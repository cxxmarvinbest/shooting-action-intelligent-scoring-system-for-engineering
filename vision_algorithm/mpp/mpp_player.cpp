
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <map>   // 用于存储核心对应的互斥锁
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/log.h>
#include <rockchip/rk_mpi.h>
#include <rockchip/rk_type.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_buffer.h>
#include <rga/rga.h>
#include <rga/RgaApi.h>
}

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace py = pybind11;

class Timer {
    std::chrono::steady_clock::time_point start;
public:
    Timer() : start(std::chrono::steady_clock::now()) {}
    double elapsed() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start).count();
    }
    void reset() { start = std::chrono::steady_clock::now(); }
};

// 获取当前时间(毫秒)
static double getTimeMs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// 全局锁映射
static std::map<int, std::mutex> g_rga_core_mutexes;
static std::mutex g_map_mutex;

static std::mutex* get_rga_core_mutex(int core) {
    std::lock_guard<std::mutex> lock(g_map_mutex);
    return &g_rga_core_mutexes[core];   // 简洁且正确
}


bool rga_convert_nv12_to_bgr(MppFrame frame, cv::Mat& out_mat, int out_width, int out_height,
                             int rga_core = RGA_NONE_CORE,
                             bool need_scale = false, int scale_w = 0, int scale_h = 0,
                             cv::Mat* scale_out = nullptr) {
    MppBuffer srcBuffer = mpp_frame_get_buffer(frame);
    if (!srcBuffer) return false;

    int srcFd = mpp_buffer_get_fd(srcBuffer);
    int width = mpp_frame_get_width(frame);
    int height = mpp_frame_get_height(frame);
    int yStride = mpp_frame_get_hor_stride(frame);
    int yHeight = mpp_frame_get_ver_stride(frame);

    if (srcFd < 0 || yStride < width || yHeight < height) return false;

    // 原图输出缓冲区
    int dstWStride = (out_width + 15) & ~15;
    int dstHStride = (out_height + 15) & ~15;
    uint8_t* dst_buf = (uint8_t*)malloc(dstWStride * dstHStride * 3);
    if (!dst_buf) return false;

    // 缩放图输出缓冲区（如果需要）
    uint8_t* scale_buf = nullptr;
    int scaleWStride = 0, scaleHStride = 0;
    bool do_scale = need_scale && scale_w > 0 && scale_h > 0 && scale_out != nullptr;
    if (do_scale) {
        scaleWStride = (scale_w + 15) & ~15;
        scaleHStride = (scale_h + 15) & ~15;
        scale_buf = (uint8_t*)malloc(scaleWStride * scaleHStride * 3);
        if (!scale_buf) {
            free(dst_buf);
            return false;
        }
    }

    // 构造 RGA 源信息（共用）
    rga_info_t src, dst;
    memset(&src, 0, sizeof(rga_info_t));
    src.fd = srcFd;
    src.mmuFlag = 1;
    src.core = rga_core;
    rga_set_rect(&src.rect, 0, 0, width, height, yStride, yHeight, RK_FORMAT_YCbCr_420_SP);

    bool ok = true;
    {
        std::mutex* mtx = get_rga_core_mutex(rga_core);
        std::lock_guard<std::mutex> lock(*mtx);

        // 1. 原图转换
        memset(&dst, 0, sizeof(rga_info_t));
        dst.virAddr = dst_buf;
        dst.mmuFlag = 1;
        dst.core = rga_core;
        rga_set_rect(&dst.rect, 0, 0, out_width, out_height, dstWStride, dstHStride, RK_FORMAT_RGB_888);
        if (c_RkRgaBlit(&src, &dst, nullptr) != 0) {
            ok = false;
            goto cleanup;
        }

        // 2. 缩放转换（如果需要）
        if (do_scale) {
            memset(&dst, 0, sizeof(rga_info_t));
            dst.virAddr = scale_buf;
            dst.mmuFlag = 1;
            dst.core = rga_core;
            rga_set_rect(&dst.rect, 0, 0, scale_w, scale_h, scaleWStride, scaleHStride, RK_FORMAT_RGB_888);
            if (c_RkRgaBlit(&src, &dst, nullptr) != 0) {
                ok = false;
                goto cleanup;
            }
        }
    }

cleanup:
    if (!ok) {
        free(dst_buf);
        if (scale_buf) free(scale_buf);
        return false;
    }

    // 原图转 BGR
    cv::Mat rgb_mat(dstHStride, dstWStride, CV_8UC3, dst_buf);
    cv::Mat bgr_mat;
    cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
    out_mat = bgr_mat(cv::Rect(0, 0, out_width, out_height)).clone();
    free(dst_buf);

    // 缩放图转 BGR（如果需要）
    if (do_scale) {
        cv::Mat rgb_scale(scaleHStride, scaleWStride, CV_8UC3, scale_buf);
        cv::Mat bgr_scale;
        cv::cvtColor(rgb_scale, bgr_scale, cv::COLOR_RGB2BGR);
        *scale_out = bgr_scale(cv::Rect(0, 0, scale_w, scale_h)).clone();
        free(scale_buf);
    } else if (scale_out) {
        scale_out->release();
    }

    return true;
}

bool rga_convert_nv12_to_rgb(MppFrame frame, cv::Mat& out_mat, int out_width, int out_height,
                             int rga_core = RGA_NONE_CORE,
                             bool need_scale = false, int scale_w = 0, int scale_h = 0,
                             cv::Mat* scale_out = nullptr) {
    MppBuffer srcBuffer = mpp_frame_get_buffer(frame);
    if (!srcBuffer) return false;

    int srcFd = mpp_buffer_get_fd(srcBuffer);
    int width = mpp_frame_get_width(frame);
    int height = mpp_frame_get_height(frame);
    int yStride = mpp_frame_get_hor_stride(frame);
    int yHeight = mpp_frame_get_ver_stride(frame);

    if (srcFd < 0 || yStride < width || yHeight < height) return false;

    int dstWStride = (out_width + 15) & ~15;
    int dstHStride = (out_height + 15) & ~15;
    uint8_t* dst_buf = (uint8_t*)malloc(dstWStride * dstHStride * 3);
    if (!dst_buf) return false;

    uint8_t* scale_buf = nullptr;
    int scaleWStride = 0, scaleHStride = 0;
    bool do_scale = need_scale && scale_w > 0 && scale_h > 0 && scale_out != nullptr;
    if (do_scale) {
        scaleWStride = (scale_w + 15) & ~15;
        scaleHStride = (scale_h + 15) & ~15;
        scale_buf = (uint8_t*)malloc(scaleWStride * scaleHStride * 3);
        if (!scale_buf) {
            free(dst_buf);
            return false;
        }
    }

    rga_info_t src, dst;
    memset(&src, 0, sizeof(rga_info_t));
    src.fd = srcFd;
    src.mmuFlag = 1;
    src.core = rga_core;
    rga_set_rect(&src.rect, 0, 0, width, height, yStride, yHeight, RK_FORMAT_YCbCr_420_SP);

    bool ok = true;
    {
        std::mutex* mtx = get_rga_core_mutex(rga_core);
        std::lock_guard<std::mutex> lock(*mtx);

        memset(&dst, 0, sizeof(rga_info_t));
        dst.virAddr = dst_buf;
        dst.mmuFlag = 1;
        dst.core = rga_core;
        rga_set_rect(&dst.rect, 0, 0, out_width, out_height, dstWStride, dstHStride, RK_FORMAT_RGB_888);
        if (c_RkRgaBlit(&src, &dst, nullptr) != 0) {
            ok = false;
            goto cleanup;
        }

        if (do_scale) {
            memset(&dst, 0, sizeof(rga_info_t));
            dst.virAddr = scale_buf;
            dst.mmuFlag = 1;
            dst.core = rga_core;
            rga_set_rect(&dst.rect, 0, 0, scale_w, scale_h, scaleWStride, scaleHStride, RK_FORMAT_RGB_888);
            if (c_RkRgaBlit(&src, &dst, nullptr) != 0) {
                ok = false;
                goto cleanup;
            }
        }
    }

cleanup:
    if (!ok) {
        free(dst_buf);
        if (scale_buf) free(scale_buf);
        return false;
    }

    // 原图直接赋值为 RGB（无需颜色转换）
    cv::Mat rgb_mat(dstHStride, dstWStride, CV_8UC3, dst_buf);
    out_mat = rgb_mat(cv::Rect(0, 0, out_width, out_height)).clone();
    free(dst_buf);

    if (do_scale) {
        cv::Mat rgb_scale(scaleHStride, scaleWStride, CV_8UC3, scale_buf);
        *scale_out = rgb_scale(cv::Rect(0, 0, scale_w, scale_h)).clone();
        free(scale_buf);
    } else if (scale_out) {
        scale_out->release();
    }

    return true;
}

class MppPlayer {
public:
    MppPlayer() : stop_flag_(false), running_(false) {
        av_log_set_level(AV_LOG_ERROR);
        if (c_RkRgaInit() != 0) {
            fprintf(stderr, "RGA 初始化失败\n");
            rga_ok_ = false;
        } else {
            rga_ok_ = true;
            rga_deinited_ = false;
        }
    }

    ~MppPlayer() {
        stop();
        if (rga_ok_ && !rga_deinited_) {
            c_RkRgaDeInit();
            rga_ok_ = false;
            rga_deinited_ = true;
        }
    }

    void set_callback_frame(py::function cb) {
        callback_frame_ = cb;
    }

    void set_callback_error(py::function cb) {
        callback_error_ = cb;
    }

    // 解码主循环函数，运行在子线程
    void decode_thread_func(const std::string& url, int display_width, int display_height,int play_fps,const bool is_wait_fps=false,const bool is_frame_drop=false,int rga_core=RGA_NONE_CORE,const bool is_mpp_scale_img=false,const int mpp_scale_w=0,const int mpp_scale_h=0) {
        auto notify_error = [&](const std::string& msg) {
            if (callback_error_) {
                py::gil_scoped_acquire gil;
                try {
                    callback_error_(msg);
                } catch (const std::exception& e) {
                    fprintf(stderr, "Error callback threw: %s\n", e.what());
                }
            }
        };
        url_=url;
        avformat_network_init();
        AVFormatContext* fmt_ctx = avformat_alloc_context();
        if (!fmt_ctx) {
            running_ = false;
            notify_error("avformat_alloc_context 失败");
            return;
        }
        // 设置中断回调
        fmt_ctx->interrupt_callback.callback = interrupt_cb;
        fmt_ctx->interrupt_callback.opaque = this;
        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "rtsp_transport", "tcp", 0);   // 用 TCP，更稳定
        av_dict_set(&opts, "fflags", "+genpts+flush_packets", 0);
        av_dict_set(&opts, "flags", "low_delay", 0);
        av_dict_set(&opts, "buffer_size", "100KB", 0);
        av_dict_set(&opts, "async", "0", 0);
        av_dict_set(&opts, "max_delay", "0.1", 0);
        av_dict_set(&opts, "skip_frame", "default", 0);
        av_dict_set(&opts, "reorder_queue_size", "10", 0); // 增加重排序队列
        // 断流/挂死保护：设置 RTSP 与底层 IO 读取超时（微秒），
        // 避免 av_read_frame 永久阻塞导致 stop() 超时 detach、MPP 队列无法释放
        av_dict_set(&opts, "stimeout", "3000000", 0);       // RTSP 读取超时 3s
        av_dict_set(&opts, "rw_timeout", "3000000", 0);     // 底层 IO 超时 3s
        printf("正在打开 RTSP: %s rga_core=%d play_fps=%d 动态等待=%d 丢帧=%d 分辨率=%dx%d 缩放=%d -> %dx%d\n", url.c_str(),rga_core,play_fps,is_wait_fps,is_frame_drop,display_width, display_height,is_mpp_scale_img,mpp_scale_w,mpp_scale_h);
        if (avformat_open_input(&fmt_ctx, url.c_str(), nullptr, &opts) != 0) {
            fprintf(stderr, "无法打开 RTSP 流\n");
            running_ = false;
            notify_error("无法打开 RTSP 流");
            avformat_free_context(fmt_ctx);   // 注意释放
            return;
        }
        av_dict_free(&opts);
//         printf("avformat_open_input 成功\n");

        int video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (video_stream < 0) {
            fprintf(stderr, "av_find_best_stream 失败，使用索引0\n");
            video_stream = 0;
        }
//         printf("使用视频流索引: %d\n", video_stream);

        MppCtx mpp_ctx = nullptr;
        MppApi* mpp_api = nullptr;
        if (mpp_create(&mpp_ctx, &mpp_api) != MPP_OK) {
            fprintf(stderr, "mpp_create 失败\n");
            avformat_close_input(&fmt_ctx);
            running_ = false;
            notify_error("mpp_create 失败");
            return;
        }
        if (mpp_init(mpp_ctx, MPP_CTX_DEC, MPP_VIDEO_CodingHEVC) != MPP_OK) {
            fprintf(stderr, "mpp_init 失败\n");
            mpp_destroy(mpp_ctx);
            avformat_close_input(&fmt_ctx);
            running_ = false;
            notify_error("mpp_init 失败");
            return;
        }
//         printf("MPP HEVC 解码器初始化成功\n");

        MppDecCfg cfg = nullptr;
        mpp_dec_cfg_init(&cfg);
        if (cfg) {
            mpp_dec_cfg_set_u32(cfg, "base:fast_out", 1);
            mpp_api->control(mpp_ctx, MPP_DEC_SET_CFG, cfg);
            mpp_dec_cfg_deinit(cfg);
        }

        MppBufferGroup frm_grp = nullptr; // 全程使用内部buffer，不接管外部ION buffer

        cv::Mat frame_mat;
        cv::Mat scale_mat;   // 新增
        AVPacket pkt;
        long long frame_count = 0;
        long long frame_count2=0;
        Timer timer;
        Timer timer2;
        double fps = 0.0;
        double where_play_fps_interval=0;
        if(play_fps>0){
            where_play_fps_interval=1000.0 /play_fps;
        }

        while (!stop_flag_) {
            double t0 = getTimeMs();
            int ret = av_read_frame(fmt_ctx, &pkt);
            if (ret < 0) {
               if (ret == AVERROR_EXIT) {   // 因中断回调触发
                    break;                  // 主动退出循环
                }
                // 如果是超时或临时错误，先检查 stop_flag_，再决定是否继续
                if (ret == AVERROR(ETIMEDOUT) || ret == AVERROR(EAGAIN)) {
                    if (stop_flag_) break;  // 停止标志已置位，直接退出
                    // 否则可选地继续等待（但建议退出并重连？这里按原逻辑继续循环）
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                if (ret == AVERROR_EOF || ret == AVERROR(EIO) || ret == AVERROR(EPIPE) || ret == AVERROR(ECONNRESET)) {
                    fprintf(stderr, "流断开 (错误码 %d)，退出当前连接\n", ret);
                    notify_error("流断开 (错误码 " + std::to_string(ret) + ")，退出当前连接");
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            if (pkt.stream_index != video_stream) {
                av_packet_unref(&pkt);
                continue;
            }
            if(stop_flag_){
                av_packet_unref(&pkt);
                break;
            }
            MppPacket mpp_pkt = nullptr;
            if (mpp_packet_init(&mpp_pkt, pkt.data, pkt.size) == MPP_OK) {
                mpp_packet_set_pts(mpp_pkt, pkt.pts);
                MPP_RET put_ret = mpp_api->decode_put_packet(mpp_ctx, mpp_pkt);
                mpp_packet_deinit(&mpp_pkt);
                if (put_ret != MPP_OK && put_ret != MPP_ERR_BUFFER_FULL) {
                    // ignore
                }
            }
            av_packet_unref(&pkt);
            if(stop_flag_){
                break;
            }
            while (true) {
                if(stop_flag_){
                    break;
                }
                MppFrame mpp_frame = nullptr;
                MPP_RET get_ret = mpp_api->decode_get_frame(mpp_ctx, &mpp_frame);
                if (get_ret != MPP_OK || !mpp_frame)
                    break;
                if(stop_flag_){
                    mpp_frame_deinit(&mpp_frame);
                    break;
                }
                if (mpp_frame_get_info_change(mpp_frame)) {
                    int w = mpp_frame_get_width(mpp_frame);
                    int h = mpp_frame_get_height(mpp_frame);
                    printf("rga_core=%d 分辨率变化: %dx%d\n", rga_core,w, h);
                    video_width_ = w;
                    video_height_ = h;
                    mpp_api->control(mpp_ctx, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
                    mpp_frame_deinit(&mpp_frame);
                    continue;
                }
                if(stop_flag_){
                    mpp_frame_deinit(&mpp_frame);
                    break;
                }
                if (mpp_frame_get_eos(mpp_frame)) {
                    mpp_frame_deinit(&mpp_frame);
                    break;
                }
                if(stop_flag_){
                    mpp_frame_deinit(&mpp_frame);
                    break;
                }

                // ---------- 动态丢帧逻辑 ----------
                ++frame_count2;
                if (play_fps > 0&&is_frame_drop&&frame_count2>1&&frame_count2%4==0) {
                    double elapsed = timer2.elapsed();
                     timer2.reset();
                    if(elapsed>0){
                        double fps2 =4.0 / elapsed;
                        if (fps2 <play_fps-4) {
                            // 丢弃当前帧，不执行 RGA 转换
                            if(print_fps_){
                                printf("rga_core=%d 丢弃当前帧，不执行 RGA 转换 elapsed=%.2f fps=%.2f < play_fps=%d \n",rga_core,elapsed,fps2,play_fps);
                            }
                            mpp_frame_deinit(&mpp_frame);
                            continue;
                        }
                    }
                }
                // ----------------------------------


                //const bool is_mpp_scale_img=false 缩放状态
                //const int mpp_scale_w=0 缩放宽度
                //const int mpp_scale_h=0 缩放高度
                //要使用MPP RGA 加速把图片缩放一份返回给模型去推理，加速时间
                bool convert_state=false;
                if(is_rgb_){
                    convert_state=rga_convert_nv12_to_rgb(mpp_frame, frame_mat, display_width, display_height,rga_core,is_mpp_scale_img,mpp_scale_w,mpp_scale_h,&scale_mat);
                }
                else{
                    convert_state=rga_convert_nv12_to_bgr(mpp_frame, frame_mat, display_width, display_height,rga_core,is_mpp_scale_img,mpp_scale_w,mpp_scale_h,&scale_mat);
                }
                if (convert_state) {
                    ++frame_count;
                    if (callback_frame_) {
                        py::gil_scoped_acquire gil;
                        try {
                            py::array_t<uint8_t> arr_src_img({frame_mat.rows, frame_mat.cols, 3}, frame_mat.data);
                            // ---------- 新增缩放处理 ----------
                            py::array_t<uint8_t> arr_scale_img;
                            if (is_mpp_scale_img && mpp_scale_w > 0 && mpp_scale_h > 0) {
                                 if (!scale_mat.empty()) {
                                    arr_scale_img = py::array_t<uint8_t>({scale_mat.rows, scale_mat.cols, 3}, scale_mat.data);
                                } else {
                                    arr_scale_img = py::array_t<uint8_t>({0, 0, 3}, nullptr);
                                }
                            } else {
                                arr_scale_img = py::array_t<uint8_t>({0, 0, 3}, nullptr);
                            }
                            callback_frame_(arr_src_img, arr_scale_img,frame_count,is_rgb_);
                        } catch (const std::exception& e) {
                            fprintf(stderr, "回调异常: %s\n", e.what());
                        }
                    }
                    if (print_fps_&&frame_count % where_print_interval_frame_ == 0) {
                        double elapsed = timer.elapsed();
                        fps = where_print_interval_frame_ / elapsed;
                        printf("rga_core=%d 已解码 %lld 帧, 当前 FPS: %.2f 当前耗时：%.2fms\n",rga_core, frame_count, fps,elapsed*1000.0/where_print_interval_frame_);
                        timer.reset();
                    }
                }
                mpp_frame_deinit(&mpp_frame);
            }
            if(stop_flag_){
                break;
            }
            //play_fps 控制一下等待时间，不然算法推理那边处理不过来
            int sleep_ms = 1;
            if(play_fps>0&&is_wait_fps){
                //判断耗时是否满足
                double t1 = getTimeMs();
                double interval=t1-t0;
                if(interval<where_play_fps_interval){
                    //动态计算等待时间
                    int sleep_ms2 = static_cast<int>(where_play_fps_interval - interval);
                    if (sleep_ms2 > 0) {
                        printf("rga_core=%d 动态计算等待时间 %d ms\n",rga_core, sleep_ms2);
                        sleep_ms=sleep_ms2;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }

        // 发送 EOS 包清空解码队列：让解码器吐光内部缓存帧，再复位释放队列
        if(mpp_ctx && mpp_api){
            MppPacket eos_pkt = nullptr;
            if (mpp_packet_init(&eos_pkt, nullptr, 0) == MPP_OK) {
                mpp_packet_set_eos(eos_pkt);
                mpp_api->decode_put_packet(mpp_ctx, eos_pkt);
                mpp_packet_deinit(&eos_pkt);
            }
            // 读取并释放所有残留帧
            MppFrame tmp_frame = nullptr;
            while (mpp_api->decode_get_frame(mpp_ctx, &tmp_frame) == MPP_OK && tmp_frame) {
                mpp_frame_deinit(&tmp_frame);
            }
            // 复位解码器，彻底释放内部队列，避免下次重新打开时队列仍处于满状态
            mpp_api->reset(mpp_ctx);
        }

        // 资源释放
        printf("frm_grp=%s\n",url.c_str());
        if (frm_grp)
            mpp_buffer_group_put(frm_grp);
        printf("mpp_ctx=%s\n",url.c_str());
        if (mpp_ctx)
            mpp_destroy(mpp_ctx);
        printf("fmt_ctx=%s\n",url.c_str());
        if (fmt_ctx)
            avformat_close_input(&fmt_ctx);
        running_ = false;
        printf("decode thread exit ok=%s\n",url.c_str());
    }

    // 非阻塞 play：启动子线程解码
    bool play(const std::string& url, int display_width = 640, int display_height = 360,int play_fps=0,const bool is_wait_fps=false,const bool is_frame_drop=false,const bool is_rgb=false,int rga_core=RGA_NONE_CORE,const bool is_mpp_scale_img=false,const int mpp_scale_w=0,const int mpp_scale_h=0) {
        if (running_) {
            if(callback_error_){
                py::gil_scoped_acquire gil;
                callback_error_("播放已启动，无法重复启动");
            }
            return false;
        }
        if (!rga_ok_) {
            fprintf(stderr, "RGA 未初始化，无法播放\n");
            if(callback_error_){
                py::gil_scoped_acquire gil;
                callback_error_("RGA 未初始化，无法播放");
            }
            return false;
        }

        stop_flag_ = false;
        running_ = true;
        is_rgb_=is_rgb;
        rga_core_=rga_core;
        // 启动独立解码线程
        decode_thread_ = std::thread(&MppPlayer::decode_thread_func, this, url, display_width, display_height,play_fps,is_wait_fps,is_frame_drop,rga_core,is_mpp_scale_img,mpp_scale_w,mpp_scale_h);
        return true;
    }

    void stop() {
        printf("stop called=%s\n",url_.c_str());
        // 始终置停止标志：触发 interrupt_cb 中断 av_read_frame 的阻塞读
        stop_flag_ = true;
        if (decode_thread_.joinable()) {
            // 等待解码线程退出（最长 10s）。interrupt_cb + stimeout/rw_timeout 应能在
            // 1~3s 内退出；若仍不退出则说明解码线程异常卡死，detach 兜底并告警。
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (decode_thread_.joinable() && std::chrono::steady_clock::now() < deadline) {
                if (!running_) {
                    break;   // 线程已退出，可安全 join
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (decode_thread_.joinable()) {
                if (!running_) {
                    decode_thread_.join();
                    printf("解码线程已 join=%s\n",url_.c_str());
                } else {
                    fprintf(stderr, "严重警告：解码线程 10s 未退出，强制 detach（MPP 队列可能未释放）=%s\n",url_.c_str());
                    decode_thread_.detach();
                }
            }
        }
        printf("stop end=%s\n",url_.c_str());
    }

    bool is_running() const { return running_; }

    void close() {
        stop();
        if (rga_ok_ && !rga_deinited_) {
            c_RkRgaDeInit();
            rga_ok_ = false;
            rga_deinited_ = true;
        }
    }

    void set_print_fps(bool print_fps,int where_print_interval_frame){
        print_fps_ = print_fps;
        where_print_interval_frame_=where_print_interval_frame;
        printf("set_print_fps: %d interval_frame=%d\n", print_fps_,where_print_interval_frame);
    }

    int get_width() const { return video_width_; }
    int get_height() const { return video_height_; }

private:
    static int interrupt_cb(void* ctx);
    std::atomic<bool> stop_flag_;
    std::atomic<bool> running_;
    bool rga_ok_ = false;
    bool rga_deinited_ = false; // 防止重复RGA deinit
    py::function callback_frame_;
    py::function callback_error_;
    bool print_fps_ = false;
    int where_print_interval_frame_=100;
    int video_width_ = 0;
    int video_height_ = 0;
    int rga_core_=0;//RGA 核心
    bool is_rgb_=false;
    std::thread decode_thread_; // 解码子线程
    std::string url_;
};

int MppPlayer::interrupt_cb(void* ctx) {
    MppPlayer* player = static_cast<MppPlayer*>(ctx);
    if (player->stop_flag_) {
        printf("interrupt_cb: stop requested\n"); // 调试时可取消注释
        return 1;
    }
    return 0;
}

PYBIND11_MODULE(mpp_player, m) {
    m.doc() = "MPP hardware decoder with RGA and Python callback (thread version)";
    py::class_<MppPlayer>(m, "MppPlayer")
        .def(py::init<>())
        .def("set_print_fps", &MppPlayer::set_print_fps, "set_print_fps")
        .def("set_callback_frame", &MppPlayer::set_callback_frame, "Set Python callback for each frame")
        .def("set_callback_error", &MppPlayer::set_callback_error, "Set callback for errors (string param)")
        .def("play", &MppPlayer::play,
             py::arg("url"),
             py::arg("display_width") = 640,
             py::arg("display_height") = 360,
               py::arg("play_fps") = 0,
                py::arg("is_wait_fps")=false,
                        py::arg("is_frame_drop")=false,
             py::arg("is_rgb")=false,
              py::arg("rga_core")=0,
              py::arg("is_mpp_scale_img")=false,
              py::arg("mpp_scale_w")=0,
              py::arg("mpp_scale_h")=0,
             "Start playing RTSP stream (non-blocking, background thread)")
        .def("stop", &MppPlayer::stop, "Stop playback")
        .def("is_running", &MppPlayer::is_running, "Check if player is running")
        .def("close", &MppPlayer::close, "Close and release all resources (RGA)")
        .def("get_width", &MppPlayer::get_width, "Get current video width (raw resolution)")
        .def("get_height", &MppPlayer::get_height, "Get current video height (raw resolution)");
}

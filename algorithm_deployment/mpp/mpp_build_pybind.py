"""
编译pybind11绑定模块的辅助脚本
在RK开发板上执行,交叉编译时请使用对应的toolchain
"""
import subprocess
import sys
import os
import shutil
import traceback


def build(toolchain=None):
    """
    编译rknn_yolov8 pybind11模块
    :param toolchain: 交叉编译toolchain文件路径, None则本地编译
    """
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    buildDir = os.path.join(scriptDir, "build")


    if os.path.exists(buildDir):
        shutil.rmtree(buildDir)
    os.makedirs(buildDir)


    try:
        cmakeCmd = ["cmake",  ".."]
        if toolchain:
            cmakeCmd.insert(1, f"-DCMAKE_TOOLCHAIN_FILE={toolchain}")

        # 执行cmake
        print("执行cmake...")
        ret = subprocess.run(cmakeCmd, cwd=buildDir)
        if ret.returncode != 0:
            print("cmake失败!")
            sys.exit(1)

        # 执行make
        print("执行make -j8...")
        ret = subprocess.run(["make", "-j8"], cwd=buildDir)
        if ret.returncode != 0:
            print("make失败!")
            sys.exit(1)

        # 复制so到python目录
        pythonDir = os.path.dirname(scriptDir)
        for f in os.listdir(scriptDir):
            if f.startswith("mpp_player") and f.endswith(".so"):
                src = os.path.join(scriptDir, f)
                dst = os.path.join(pythonDir, f)
                shutil.copy2(src, dst)
                print(f"已复制: {src} -> {dst}")

        print("编译完成!")

    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolchain", default=None, help="交叉编译toolchain路径")
    args = parser.parse_args()
    build(args.toolchain)

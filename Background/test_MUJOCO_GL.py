"""
测试 MUJOCO_GL 渲染后端是否可用。
支持三种模式：egl（无头GPU）、osmesa（CPU软渲染）、glfw（有窗口）
用法：
    python test_mujoco_gl.py            # 默认用 egl
    python test_mujoco_gl.py osmesa     # 用 osmesa
    python test_mujoco_gl.py glfw       # 用 glfw（需要显示器）
"""

import sys
import os

# ---------- 1. 设置渲染后端 ----------
GL_BACKEND = sys.argv[1] if len(sys.argv) > 1 else "egl"
os.environ["MUJOCO_GL"] = GL_BACKEND
print(f"[INFO] MUJOCO_GL = {GL_BACKEND}")

# ---------- 2. 导入 MuJoCo ----------
try:
    import mujoco
    print(f"[OK]   mujoco 版本: {mujoco.__version__}")
except ImportError as e:
    print(f"[FAIL] 无法导入 mujoco: {e}")
    sys.exit(1)

import numpy as np

# ---------- 3. 构建一个最简单的 MuJoCo 模型（单摆） ----------
XML = """
<mujoco model="test">
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
    <body name="pendulum" pos="0 0 1">
      <joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05" fromto="0 0 0  0 0 -0.5" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

try:
    model = mujoco.MjModel.from_xml_string(XML)
    data  = mujoco.MjData(model)
    print("[OK]   MjModel / MjData 创建成功")
except Exception as e:
    print(f"[FAIL] 模型创建失败: {e}")
    sys.exit(1)

# ---------- 4. 仿真几步 ----------
try:
    for _ in range(10):
        mujoco.mj_step(model, data)
    print("[OK]   仿真步进正常")
except Exception as e:
    print(f"[FAIL] 仿真步进失败: {e}")
    sys.exit(1)

# ---------- 5. 渲染一帧（关键测试） ----------
WIDTH, HEIGHT = 640, 480
try:
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    renderer.update_scene(data)
    pixels = renderer.render()          # numpy array (H, W, 3)
    renderer.close()

    assert pixels.shape == (HEIGHT, WIDTH, 3), f"像素shape异常: {pixels.shape}"
    assert pixels.dtype == np.uint8,           f"像素dtype异常: {pixels.dtype}"
    print(f"[OK]   渲染成功，图像 shape={pixels.shape}, dtype={pixels.dtype}")
    print(f"       像素均值={pixels.mean():.1f}  (非全黑说明渲染正常)")
except Exception as e:
    print(f"[FAIL] 渲染失败: {e}")
    sys.exit(1)

# ---------- 6. 可选：保存图像 ----------
try:
    from PIL import Image
    img = Image.fromarray(pixels)
    out_path = f"test_mujoco_gl_{GL_BACKEND}.png"
    img.save(out_path)
    print(f"[OK]   图像已保存到 {out_path}")
except ImportError:
    print("[SKIP] Pillow 未安装，跳过图像保存（pip install Pillow）")

print("\n✅ MUJOCO_GL 测试全部通过！")

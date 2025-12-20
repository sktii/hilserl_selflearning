import time
import gymnasium as gym
import numpy as np
import cv2
import imageio

# 1. 匯入您的環境與設定
from franka_sim.envs.panda_stack_gym_env import PandaStackCubeGymEnv
from examples.experiments.stack_cube_sim.config import EnvConfig

def main():
    # 2. 初始化設定 (這會包含您定義的 left/right/wrist 相機參數)
    config = EnvConfig()

    # 3. 直接實例化環境 (比 gym.make 更穩定，且確保用到正確的 class)
    #    這裡我們強制傳入 config，解決 AttributeError
    env = PandaStackCubeGymEnv(
        render_mode="human", 
        image_obs=True, 
        config=config
    )

    action_spec = env.action_space

    def sample():
        a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
        return a.astype(action_spec.dtype)

    obs, info = env.reset()
    frames = []

    print("開始渲染測試... (請檢查彈出的視窗是否包含 Left, Right, Wrist 畫面)")

    for i in range(200):
        a = sample()
        obs, rew, done, truncated, info = env.step(a)
        
        # 4. 取得影像並拼接顯示
        # 根據您的 Config，現在應該有 left, right, wrist
        images = obs["images"]
        
        # 動態抓取所有相機的畫面並橫向拼接
        # 確保順序一致 (例如 left -> wrist -> right)
        camera_keys = ["left", "wrist", "right"] 
        # 如果 config 裡面的 key 不一樣，這裡會自動過濾
        valid_keys = [k for k in camera_keys if k in images]
        
        if valid_keys:
            img_list = [images[k] for k in valid_keys]
            concat_img = np.concatenate(img_list, axis=1) # 橫向拼接
            
            # OpenCV 顯示 (按 'q' 離開)
            # 注意: gym 輸出通常是 RGB，OpenCV 需要 BGR，所以轉一下顏色
            # cv2.imshow("Camera Views (Left | Wrist | Right)", cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            frames.append(concat_img)
        else:
            print("警告: 觀測資料中找不到影像，請檢查 Config 設定。")

        if done:
            obs, info = env.reset()

    env.close()
    cv2.destroyAllWindows()

    # 5. 儲存影片 (可選)
    if frames:
        print(f"正在儲存測試影片: franka_stack_test.mp4 ...")
        imageio.mimsave("franka_stack_test.mp4", frames, fps=20)
        print("影片儲存完成。")

if __name__ == "__main__":
    main()
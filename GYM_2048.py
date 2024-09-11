# train_dqn.py
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np
from env import Game2048Wrapper

# Callback để lưu loss
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.losses = []

    def _on_step(self) -> bool:
        # Lưu loss sau mỗi bước huấn luyện
        if 'loss' in self.model.logger.name_to_value:
            self.losses.append(self.model.logger.name_to_value['loss'])
        return True

# Tạo môi trường
env = Game2048Wrapper()

# Tạo mô hình DQN (Double DQN mặc định)
model = DQN('MlpPolicy', env, verbose=1, policy_kwargs={'net_arch': [256, 256]})

# Tạo callback để lưu loss
callback = SaveOnBestTrainingRewardCallback(check_freq=1000)

# Huấn luyện mô hình và lưu điểm số, giá trị ô lớn nhất
scores = []
max_tiles = []

for i in range(10):  # Số lượng màn chơi để thống kê, giảm số lượng để kiểm tra
    print(f"Playing game {i+1}")
    obs = env.reset()
    score = 0
    max_tile = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        score += rewards
        max_tile = max(max_tile, env.game.max_tile())
        if done:
            break
    scores.append(score)
    max_tiles.append(max_tile)
    print(f"Game {i+1} finished with score {score} and max tile {max_tile}")

# Huấn luyện mô hình với callback
print("Starting training...")
model.learn(total_timesteps=10000, callback=callback)  # Giảm tổng số bước huấn luyện để kiểm tra

# Lưu mô hình
model.save("dqn_2048")

# Vẽ biểu đồ
plt.figure(figsize=(18, 5))

# Biểu đồ Điểm Số Thống Kê Tương Ứng với Số Lượng Màn Chơi
plt.subplot(1, 3, 1)
plt.plot(scores)
plt.xlabel('Số lượng màn chơi')
plt.ylabel('Điểm số')
plt.title('Biểu đồ Điểm Số Thống Kê')

# Biểu đồ Số Lượng Màn Chơi Có Ô Giá Trị Lớn Nhất Tương Ứng
plt.subplot(1, 3, 2)
plt.plot(max_tiles)
plt.xlabel('Số lượng màn chơii')
plt.ylabel('Giá trị ô lớn nhất')
plt.title('Biểu đồ Số Lượng Màn Chơi Có Ô Giá Trị Lớn Nhất Tương Ứng')

# Biểu đồ Loss theo thời gian
plt.subplot(1, 3, 3)
plt.plot(callback.losses)
plt.xlabel('Thời gian')
plt.ylabel('Loss')
plt.title('Biểu đồ Loss theo thời gian')

plt.tight_layout()
plt.show()

# Lưu biểu đồ
plt.savefig("training_results.png")

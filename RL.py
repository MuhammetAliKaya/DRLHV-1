import gym
import numpy as np
import random
import time
import pygame
import os
import matplotlib.pyplot as plt 

# files for save q-table
npy_file = "q_table.npy"
csv_file = "q_table.csv"

env = gym.make("Taxi-v3")
action_size = env.action_space.n
state_size = env.observation_space.n

state_message = "VAR" if os.path.exists(npy_file) else "YOK"
print(f"Mevcut Q-Table dosyası durumu: {state_message}")
decision = input("Yeniden Eğitim Yapılsın mı? (e/h): ").lower()

q_table = None

if decision == "e":
    print("Eğitim modu seçildi. Sıfırdan başlanıyor...")
    q_table = np.zeros((state_size, action_size))

    # important paramaters diret effect on models learning
    num_episodes = 5000
    max_steps = 99
    learning_rate = 0.7
    discount_rate = 0.618
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01

    rewards_history = []
    epsilon_history = []

    print(f"Eğitim Başlıyor... ({num_episodes} bölüm)")
    # training
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        for step in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            new_state, reward, done, truncated, info = env.step(action)

            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward
                + discount_rate * np.max(q_table[new_state, :])
                - q_table[state, action]
            )
            total_reward += reward
            state = new_state
            if done:
                break
        rewards_history.append(total_reward)
        epsilon_history.append(epsilon)
        
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode
        )
        # info about trainings complate rate
        if (episode + 1) % 500 == 0:
            print(f"Bölüm {episode + 1} tamamlandı...")

    print("Eğitim Bitti!")
    # save locally train output
    np.save(npy_file, q_table)
    print(f"NPY '{npy_file}' kaydedildi.")

    print("Grafikler oluşturuluyor...")
    
    # 1. Grafik: Öğrenme Eğrisi (Rewards)
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history)
    plt.title('Öğrenme Eğrisi (Bölüm Başına Toplam Ödül)')
    plt.xlabel('Bölüm (Episode)')
    plt.ylabel('Toplam Ödül')
    plt.savefig('training_rewards.png') 
    print(" - training_rewards.png kaydedildi.")
    
    # 2. Grafik: Epsilon Azalması
    plt.figure(figsize=(12, 6))
    plt.plot(epsilon_history, color='orange')
    plt.title('Epsilon Decay (Keşif Oranı)')
    plt.xlabel('Bölüm (Episode)')
    plt.ylabel('Epsilon Değeri')
    plt.savefig('epsilon_decay.png')
    print(" - epsilon_decay.png kaydedildi.")
    plt.close('all')
    # ------------------------------------------

    headers = "Guney(0), Kuzey(1), Dogu(2), Bati(3), YolcuAl(4), YolcuBirak(5)"
    np.savetxt(csv_file, q_table, delimiter=",", header=headers, fmt="%.4f")
    print(f"İnsan için okunabilir '{csv_file}' oluşturuldu.")

else:
    # run with trained data
    if os.path.exists(npy_file):
        print(f"'{npy_file}' dosyası yükleniyor...")
        q_table = np.load(npy_file)
        print("Yükleme başarılı.")
    else:
        print("Dosya yok, mecburen boş tabloyla devam edilecek")
        q_table = np.zeros((state_size, action_size))


print("-" * 30)
input("İzlemek için ENTER'a bas...")
try:
    if "env" in locals():
        env.close()

    env = gym.make("Taxi-v3", render_mode="human")

    def smart_wait(second):
        start = time.time()
        while time.time() - start < second:
            pygame.event.pump()
            time.sleep(0.1)

    # console output visualised tests
    for i in range(5):
        state, info = env.reset()

        env.render()
        smart_wait(1.0)

        done = False
        total_reward = 0
        step_count = 0
        max_test_steps = 50

        print(f"--- TUR {i + 1} BAŞLIYOR ---")

        while not done:
            pygame.event.pump()

            action = np.argmax(q_table[state, :])
            new_state, reward, done, truncated, info = env.step(action)

            state = new_state
            total_reward += reward
            step_count += 1

            time.sleep(0.3)

            if step_count > max_test_steps:
                print(f"UYARI: Taksi {max_test_steps} hamlede bitiremedi! (Kısırdöngü)")
                break

        print(f"Tur Bitti. Puan: {total_reward}")
        smart_wait(1.0)

    print("Tüm turlar bitti. Kapatılıyor...")
    env.close()
# reprot any problem
except Exception as e:
    print("Hata:", e)
    import traceback

    traceback.print_exc()

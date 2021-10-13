import gym
import numpy as np
from numpy.lib.npyio import mafromtxt
from processing_py import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.agents.dqn import DQNAgent

makernd = lambda a,b:np.int(np.random.random_sample()*(b-a)+a)

class MyEnv(gym.Env):
    def __init__(self):
        self.app = App(600,400)
        self.reset()
        self.actions=np.array([0,20,-20])
    def reset(self):
        self.timer=1.0
        self.ball_x_speed=5
        if makernd(0,2)==0:
            self.ball_x_speed=-5
        self.ball_y_speed=5
        self.blocks=np.array([1]*10)
        self.score=0
        self.racket_x=300
        self.ball_x=makernd(0,590)
        self.ball_y=makernd(110,150)
        self.observation=np.hstack([self.blocks,self.ball_x,self.ball_y,self.ball_x_speed,self.ball_y_speed,self.racket_x])
        self.remain=1.0
        return self.observation
    def step(self,action):
        if self.timer>0:
            self.timer-=0.000001
        else:
            #pass
            print("timer is 0")
        r=0.0
        #ラケットが壁から出ないようにする
        if self.racket_x<=0 and action==2:
            pass
        elif self.racket_x>=540 and action==1:
            pass
        else:
            self.racket_x+=self.actions[action]
        self.ball_x+=self.ball_x_speed
        self.ball_y+=self.ball_y_speed
        self.observation=np.hstack([self.blocks,self.ball_x,self.ball_y,self.ball_x_speed,self.ball_y_speed,self.racket_x])
        #ボールがラケットに当たったかどうか
        if self.ball_x>=self.racket_x and self.ball_x<=self.racket_x+60 and self.ball_y<=350 and self.ball_y>=340:
            self.ball_y_speed=-self.ball_y_speed
        #ボールがブロックに当たったかどうか
        for i in range(10):
            if self.blocks[i]==1 and self.ball_x>=i*60 and self.ball_x<=i*60+60 and self.ball_y>=50 and self.ball_y<=110:
                self.blocks[i]=0
                self.ball_y_speed=-self.ball_y_speed
                self.remain+=0.1
                r+=5
                r*=self.remain
        ###
        if self.ball_x<0 or self.ball_x+10>600:
            self.ball_x_speed=-self.ball_x_speed
        if self.ball_y<=0:
            self.ball_y_speed=-self.ball_y_speed
        ###
        if  self.ball_y+10>=400:
            return self.observation,np.float32(-50),True,{} #失敗
        elif self.is_game_clear():
            return self.observation,np.float32(r+100),True,{} #成功
        else:
            if abs(self.ball_x-self.racket_x-25)>=60:
                r-=0.1
            else:
                r+=0.1
            if r>0:
                r*=self.timer
            return self.observation,np.float32(r),False,{} #ブロックにヒットしたとき点数を与える



    #ブロックが全部消えたかどうかを調べる
    def is_game_clear(self):
        for i in range(10):
            if self.blocks[i]==1:
                return False
        return True



    def render(self,mode):
        self.app.background(10,10,10)
        self.draw_blocks()
        self.draw_racket()
        self.draw_ball()
        self.app.redraw()
    def draw_blocks(self):
        for i in range(10):
            if self.blocks[i]==1:
                self.app.rect(i*60,50,60,60)
    def draw_racket(self):
        self.app.rect(self.racket_x,350,60,10)
    def draw_ball(self):
        self.app.circle(self.ball_x,self.ball_y,10)

if __name__=='__main__':
    env=MyEnv()
    env.reset()
    model = Sequential([Flatten(input_shape=(1,15)),Dense(16,activation='relu'),Dense(16,activation='relu'),Dense(16,activation='relu'),Dense(3,activation='linear')])
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model,nb_actions=3,gamma=0.99,memory=memory,nb_steps_warmup=100,target_model_update=1e-2,policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(env,nb_steps=100000,visualize=True,verbose=1)                           #visualize=Falseにすれば、画面描写をしなくなる
    #
    # dqn.model.save('game',overwrite=True)
    dqn.test(env,nb_episodes=10,visualize=True)
    

import argparse
import numpy as np
from tqdm import tqdm
import glob
import pandas as pd
import time

class Environment:

    def __init__(self, path: str):

        #-----------------------------------------------------------------------
        # 데이터 파일을 읽는다.
        #-----------------------------------------------------------------------
        f = open(path, 'r')
        lines = f.readlines()
        f.close()

        for i in range(len(lines)):
            lines[i] = lines[i].split()
    
        # 첫번째 라인: 무기의 개수, 타겟의 개수, 무기당 포탄의 개수
        self.m = int(lines[0][0])       # 무기의 개수
        self.n = int(lines[0][1])       # 타겟의 개수
        self.b = int(lines[0][2])       # 무기 당 포탄의 개수
        self.mu = [self.b] * self.m 
        
        # 다음 n개의 라인: 타겟의 중요도
        self.w = []
        for line in lines[1:self.n+1]:
            self.w.append(int(line[0]))
        self.w = np.array(self.w)

        # 이후의 라인: 파괴율
        p = np.zeros((self.m, self.n))
        for line in lines[self.n+1:]:
            i, j, k = int(line[0]), int(line[1]), float(line[2])
            p[i,j] = k     # 타겟에 대한 무기의 파괴율
        self.p = p

        # 타겟의 내구도 및 포탄의 개수를 초기화
        self.reset()

    def reset(self):
        self.durability = np.ones((self.n))
        self.mu = [self.b] * self.m

    def AssignWeaponsToTargetsGreedy(self):
        wta = np.zeros((self.m, self.b))
        eff = (self.p * self.w).flatten()

        if args.verbose == False: pbar = tqdm(total=self.m*self.b)
        while np.max(self.mu) > 0:

            # 포탄이 1개 이상 있으면 1, 아니면 0인 리스트 b를 만든다.
            bullets_left = np.copy(self.mu)
            bullets_left[bullets_left>=1] = 1
            b = np.repeat(bullets_left, self.n)   
            
            # 가장 포격의 효과가 높은 무기-타겟 페어를 선택한다. (그리디 방식)
            ind = np.argmax(eff*b)
            row = ind // self.n
            col = ind % self.n
            
            #indices = np.argsort(eff*b)[::-1]
            #row = indices[0] // self.n
            #col = indices[0] % self.n

            # 할당결과를 기록한다.
            wta[row, self.b - self.mu[row]] = int(col)
            
            if args.verbose == True: print(" %3d번 무기로 %3d번 타겟을 공격합니다."%(row, col))

            # 남은 포탄의 수를 업데이트한다.
            self.mu[row] -= 1
            
            # 타겟의 내구도를 업데이트한다.
            self.durability[col] = self.durability[col] * (1 - self.p[row][col])
            
            # eff를 업데이트한다. (한번 포격한 타겟은 포격 효과가 떨어짐)
            update_indices = np.arange(self.m) * self.n + col
            eff[update_indices] *= self.durability[col]

            if args.verbose == False: pbar.update(1)
       
    def GetConfiguration(self):
        return self.m, self.n, self.b

    def GetScore(self):
        return np.sum((np.multiply(self.durability, self.w)))


def main(args):

    envFiles = glob.glob('data/*.txt')
    df = pd.DataFrame(columns=['name', 'weapons', 'targets', 'bullets', 'score', 'elapsed'])
    
    for fn in envFiles:
        env = Environment(fn)
        fn2 = fn.split('/')[1]
        fn3 = fn2[4:-4]
        time_start = time.time()
        env.AssignWeaponsToTargetsGreedy()
        time_elapsed = time.time() - time_start
        weapons, targets, bullets = env.GetConfiguration()
        score = env.GetScore()
        df.loc[len(df)] = [fn3, weapons, targets, bullets, score, time_elapsed]

    df.to_csv('result.csv', index=False)
        
    
if __name__ == '__main__':

    #---------------------------------------------------------------------------
    # command line argument를 파싱합니다.
    #---------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='랜덤 넘버 시드')
    parser.add_argument('--debug', action='store_true', help='디버그 모드 사용')
    parser.add_argument('--envPath', default='data/wta_500x1000x1.txt', type=str, help='데이터 경로')
    parser.add_argument('--verbose', action='store_true', help='무기-타겟 할당 출력')
    args = parser.parse_args()

    #---------------------------------------------------------------------------
    # 디버그 모드 진입
    #---------------------------------------------------------------------------
    if args.debug:
        debugpy.listen(5678)
        print("디버거를 실행시켜주십시오.")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print("첫 브레이크포인트입니다.")

    #---------------------------------------------------------------------------
    # 랜덤 넘버 시드 적용
    #---------------------------------------------------------------------------
    seed = args.seed
    np.random.seed(seed)

    #---------------------------------------------------------------------------
    # 메인 함수 호출
    #---------------------------------------------------------------------------
    main(args)






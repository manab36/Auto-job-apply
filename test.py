from tqdm import tqdm
import time
for i in tqdm(range(100), dynamic_ncols=True):
    time.sleep(0.2)
    

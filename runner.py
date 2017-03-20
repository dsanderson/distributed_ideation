import subprocess
import os
import random
import re
import statistics
import tqdm

def resample(size, sample, out_file):
    #reseample for the bootstrapping
    sample_inds = [random.randrange(0,size) for _ in range(0,size)]
    sample = [sample[i] for i in sample_inds]
    with open(out_file,'w',encoding="latin1") as f:
        f.write("Unique ID,Text: Verbatim\n")
        f.write("\n".join(sample))

def subsample(size, in_file):
    with open(in_file, "r", encoding="latin1") as f:
        lines = f.readlines()
    lines = [l for l in lines[1:] if l.strip()!=""]
    random.shuffle(lines)
    return lines

def run_analyze(k, size, seed):
    res = subprocess.run(["python3", "analyze.py", str(k), str(size), str(seed)], stdout=subprocess.PIPE)
    txt = res.stdout
    m = re.search(b"num_clusters:(\d+)",txt)
    num_clusters = m.group(0)[len("num_clusters:"):]
    num_clusters = int(num_clusters)
    m = re.search(b"dev_clusters:(\d+)",txt)
    dev_clusters = m.group(0)[len("dev_clusters:"):]
    dev_clusters = int(dev_clusters)
    return (num_clusters, dev_clusters)

def print_results(points):
    for p in points:
        print(p)

if __name__ == '__main__':
    #capture the environment variables
    k_min = int(os.environ['K_MIN'])
    k_max = int(os.environ['K_MAX'])
    seed = int(os.environ['SEED'])
    size = int(os.environ['SIZE'])
    random.seed(seed)
    sample = subsample(size, "raw_samples.csv")
    points = []
    resample(len(sample), sample, "samples.csv")
    for k in tqdm.tqdm(range(k_min,k_max+1)):
        res = run_analyze(k, size, seed)
        points.append((size, k, res[0], res[1]))
    print_results(points)

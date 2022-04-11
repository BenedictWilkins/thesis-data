from thesisdata.utils._omegaconf import symlinked

if __name__ == "__main__":
    symlinked("~/.data/WOB/train", ["~/.data/WOB/world-of-bugs-normal/NORMAL-TRAIN", "~/.data/WOB/world-of-bugs-normal/NORMAL-TRAIN-SMALL"])
    
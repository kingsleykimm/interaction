# interaction tasks

Just a sandbox right now to test AIHabitat.

# Data collection:
- Include information about required habitat assets, as well as the configs
- Include


Data labelling done with LLava-NeXT-Video-Qwen2-32B, clone the LLaVa-NeXT repo and put it in the root repo
run the data_annotation bash script inside the dataset folder

TO-DO: Think about using an outer loop to take in multiple scenes / instances to run data_collection on

Common issues that I run into: 
- the vision tower in LLaVa-Next is not working that well, keep freezing at 12/14, what was the fix before?
    Solution: module load cuda/12.2.2 apptainer pytorch/2.0.1
    apptainer run --nv $CONTAINERDIR/pytorch-2.0.1.sif
    inference needs to be run on appropriate GPU sizes (64GB) for 32
- seems like auto_batch_size in model_config is the way to go
Common habitat issues:
- Make sure you are never accessing env variables before calling env.reset(), and intending to use them after the env.reset()

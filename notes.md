# VPoser Notes

Vposer is just a VAE that gets a prior on the distribution of human 3d poses, so once we have a pretrained version, it's good to just few in noise vectors / possible poses and get a new pose. However, that's just one part. The IK engine is what will actually help give us fine-grained control over the poses, need to look into that.

## IK Engine
- has a vposer
- num betas?

the forward function:

source_kpts is a function that given body parameters computes source key points that should match target key points
. Try to reconstruct the bps signature by optimizing the body_poZ

so is the ik engine fitting to the body model it's fed in?

## Body Model:
you feed in the *.pkl motion file here and it basically does all the processing, in the forward function when you pass in the poses it will calculate the mesh and the next joints for you
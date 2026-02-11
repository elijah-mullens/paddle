<div class="alert alert-info">

### Define Distribute Properties ($\texttt{distribute}$)

The seventh dictionary entry, $\texttt{distribute}$, describes how parallel computing is accomplished (and therefore only important for running py scripts in terminal). We will still go over the properties here since it will be pertinent later in the tutorial:

</div>

```
# Two choices for backend
# 'gloo' for CPU parallel computing
# 'nccl' for GPU
backend = 'gloo'

# Two choices for layout
# 'slab' for cartesian environments
# 'cubed-sphere' for 3D spheres
layout = 'slab'

# nb2 and nb3 are the number of blocks in each dimension
# These two numbers determine the amount of cores you can use
# For slab layouts, number of cores = nb2 * nb3
# For jupyter notebooks, we can only use 1 core in a cell
# We will go over multi-core processing more explicitly at the end of this notebook
nb2 = 1
nb3 = 1

# verbose
# Running the code will always produce time-stepping outputs to terminal
# The verbose here is explicitly for printing out information useful for debugging
verbose = False
```

Now we create the dictionary object for the future .yaml file

```
# Make Dictionary
distribute_dict = { 'backend': backend,
                    'layout': layout,
                    'nb2': nb2,
                    'nb3': nb3,
                    'verbose': verbose}
```

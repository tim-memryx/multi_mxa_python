# Separate Apps, Separate DFPs, Separate MXAs

Here's an example of running MiDAS Depth + Cartoonizer on separate M.2s, each in their own processes.


## Requirements

Make sure you've activated your memryx python venv and have installed the `opencv-python` pip package.


## Run

Start each example in their own terminal windows, and make sure to have the venv activated.

### Start Depth Estimation

```python
python3 midas.py -id 0 -x 0
```

This will use camera 0 (`-id` argument) and MXA 0 (`-x` argument) to run Depth Estimation.

### Start Depth Estimation

Next, in a separate terminal, run:

```python
python3 cartoonizer.py -id 2 -x 1
```

This will use camera 2 (`-id` argument) and MXA 1 (`-x` argument) to run the Cartoonizer style transfer demo.

**Note**: Linux indexes webcams as 0,2,4,6,etc. So your first cam is 0, second cam is 2, third cam is 4, and so on.


## How Do They Work?

By changing the `group_id` parameter to the [AsyncAccl](https://developer.memryx.com/api/accelerator/python.html) constructor, we indicate which device to use.

Simply starting separate apps that use different `group_id`s will be enough to run them fully in parallel.

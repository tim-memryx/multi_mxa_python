# Same DFP Across 4 MXAs

This example runs the input camera stream or input video file robin-robin across **all 4** connected M.2 modules.


## Requirements

Make sure you've activated your memryx python venv and have installed the `opencv-python` pip package.


## Run

Camera example (cam ID 0):

```python
python3 run.py -id 0
```

Video file example:

```python
python3 run.py -f my_video.mp4
```


## How Does This Work?

Same as before, to enable the apps to run in parallel, we are simply setting the `group_id` argument for AsyncAccl to the index of the MXA M.2 module we want to use.

Now in the main image capture thread, we are pushing round-robin to 4 different object input queues in order.

The display thread then reads from the objects, also round-robin and in order.

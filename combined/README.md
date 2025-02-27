# Combined Application using Different DFPs on Separate M.2s

**Please read the Separate Processes example first!**

Here's an example of running MiDAS Depth + Cartoonizer on separate M.2s, but within a single application that uses multiple AsyncAccl objects.

It uses a single input camera and concats app outputs together.


## Requirements

Make sure you've activated your memryx python venv and have installed the `opencv-python` pip package.


## Run

```python
python3 run.py -id 0
```

This will use MXA 0 for Depth and MXA 1 for Cartoonizer. Edit run.py if you want to try different M.2s in your system.



## How Do They Work?

Same as in separate processes, to enable the apps to run in parallel, we are simply setting the `group_id` argument for [AsyncAccl](https://developer.memryx.com/api/accelerator/python.html) to the index of the MXA M.2 module we want to use.

In this case, we're using two AsyncAccl objects within the same Python application, instead of two completely separate applications.

We use a single camera & display thread to operate on the input stream.

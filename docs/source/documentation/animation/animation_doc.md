## Use Case

The `Animator` class offers some simple functionality to create animations in multiple video formats (e.g., .gif and .mp4). Specific `Animator` classes have been created to help create specific visual prompts, such as bar plots and cartesian plots. These prompts are primarily used for training of regression-based myoelectric control systems, but the `Animator` class can also be used to generate these prompts for any purpose. For fully custom animations, you can inherit from the `Animator` class and pass a set of frames to `save_video()` (see other `Animators` for examples). If you'd rather implement a custom plotting animation that isn't supported, you can inherit from `PlotAnimator` instead.

Every `Animator` takes an `output_filepath` (default `'libemg.gif'`) and `fps` (default 24). The video format is chosen from the file extension, and both `.gif` and `.mp4` are supported. The plot animators create the video with `save_plot_video(coordinates)`, where `coordinates` is an N x M array with one row per frame and one column per degree of freedom. Pass `save_coordinates=True` to also save the coordinates to a `.txt` file next to the video (e.g., as ground truth labels for regression).

For all further examples, we will use a generated set of coordinates to illustrate the difference between plots.

```Python
import numpy as np

fps = 24
coordinates = np.concatenate((
    np.linspace(0, 1, num=fps),  # each movement is 24 frames -> 1 second
    np.ones(2 * fps),   # steady state
    np.linspace(1, -1, num=2 * fps),
    np.ones(2 * fps) * -1,
    np.linspace(-1, 0, num=fps)
))
coordinates = np.hstack((
    np.expand_dims(coordinates, 1),
    np.zeros((coordinates.shape[0], 1))
))
```

## Bar Plots

One type of `Animator` built into `LibEMG` creates bar plot animations, with one bar (and label) per degree of freedom. Pass in an array of coordinates to the `save_plot_video` method to create a bar plot visualization (see Figure 1).

```Python
from libemg.animator import BarPlotAnimator

animator = BarPlotAnimator(['Open', 'Close'], fps=fps)
animator.save_plot_video(coordinates)
```

![alt text](bar.gif)
<p align="center">Figure 1: Simple bar plot animation.</p>

Additional information can also be shown during these animations, such as the next destination and a countdown for steady states (see Figure 2).

```Python
animator = BarPlotAnimator(['Open', 'Close'], fps=fps, show_countdown=True, show_direction=True)
animator.save_plot_video(coordinates)
```

![alt text](bar-info.gif)
<p align="center">Figure 2: Bar plot animation with added information.</p>

Parameters such as the time per unit distance, figure size, and more can also be modified. See the `BarPlotAnimator` API for more details. `SingleDirectionBarPlotAnimator` works the same way, but only shows positive values (0 to 1).

## Scatter Plots

`LibEMG` also provides a helper class to animate scatter plots, where the first two columns of `coordinates` are the x and y positions of the point (see Figure 3). Unlike the bar plot, the scatter plot does not take labels; images can instead be placed around the plot with `axis_images`.

```Python
from libemg.animator import ScatterPlotAnimator

animator = ScatterPlotAnimator(fps=fps)
animator.save_plot_video(coordinates)
```

![alt text](scatter.gif)
<p align="center">Figure 3: Simple scatter plot animation.</p>

Similar to the bar plot animation, extra information can be added such as next destination, a countdown, and a unit circle boundary (see Figure 4).

```Python

animator = ScatterPlotAnimator(fps=fps, show_countdown=True, show_direction=True, show_boundary=True)
animator.save_plot_video(coordinates)
```

![alt text](scatter-info.gif)
<p align="center">Figure 4: Scatter plot animation with added information.</p>

Parameters such as the time per unit distance, figure size, and more can also be modified. See the `ScatterPlotAnimator` API for more details.

## Other Cartesian Plots

`ArrowPlotAnimator` and `TargetPlotAnimator` animate a third degree of freedom on top of the x and y position, so they expect three columns in `coordinates` (values from -1 to 1). The third column sets the angle of the arrow for `ArrowPlotAnimator` and the radius of the target for `TargetPlotAnimator`. Both take the same arguments as `ScatterPlotAnimator` (except `plot_line`).

# Soccer Bot
A soccer bot that uses a basic CNN to detect the ball and click on it.
![Main UI]("assets/screenshots/main_ui.jpg")

## Instructions
Execute the following command to align your game window to the correct position. Follow the on screen instructions to exit/resume/pause the bot.
```python
python3 soccer.py --preview
```

Execute the following command to run without the preview window.
```python
python3 soccer.py
```

## Performance
- Gets 60ms per frame on a FX6300 (Hexacore) @ 4.2GHz
- Model has an IOU of 0.81 on training data
- Model doesn't know when ball has exited the frame and spams emotes

## Todo
- Reduce number of outputs to 2 (x_centre, y_centre)
- Use a residual network using Keras' functional api
- Implement a confidence score to detect when ball isn't present on screen
- Get model to work on even lower end cpu

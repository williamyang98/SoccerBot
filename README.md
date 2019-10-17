# Soccer Bot
A soccer bot that uses a basic CNN to detect the ball and click on it.
![alt text](docs/main_ui.png "Main UI")

## Installation
Download the repository either as a zip or through git
```
git clone https://github.com/FiendChain/SoccerBot
```
Install python3 if you havent already. Then install the requirements through the following command.
```
pip install -r requirements.txt
```

## Instructions
Execute the following command to align your game window to the correct position. Follow the on screen instructions to exit/resume/pause the bot.
```
python3 soccer.py --preview
```

Execute the following command to run without the preview window.
```
python3 soccer.py
```

## Performance
- Gets 30ms per frame on a i5-4200u (mobile) 
- Model has an IOU of 0.89 on training data
- Model doesn't know when ball has exited the frame and spams emotes
- Can essentially get an infinite score

## Todo
- Reduce number of outputs to 2 (x_centre, y_centre)
- Use a residual network using Keras' functional api
- Implement a confidence score to detect when ball isn't present on screen

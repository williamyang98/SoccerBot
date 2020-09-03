# Soccer Bot
A soccer bot that uses a basic CNN to detect the ball and click on it.

### Youtube Video
[![SoccerBot](http://img.youtube.com/vi/zDZrXnTsxvo/0.jpg)](http://youtu.be/zDZrXnTsxvo "SoccerBot")

### UI
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
python3 soccer.py
```

Since Facebook has removed the messenger soccer app, there is an python clone of the game. You can run this using the following command.
```
python3 emulator.py
```

## Performance
#### Large model (100,000 parameters - 117 kB quantized)
- Gets 30ms per frame on a i5-7200u (30fps) (laptop cpu) 
- Gets 30ms per frame on a fx-6350 (30fps) (desktop cpu) 
- Can essentially get an infinite score

#### Small model (30,000 parameters - 51kB quantized)
- Gets 15ms per frame on a i5-7200u (60fps) (laptop cpu) 
- Gets 15ms per frame on a fx-6350 (60fps) (desktop cpu) 
- Can essentially get an infinite score 
- Slightly lower model accuracy compared to large model

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
- Gets 30ms per frame on a i5-7200u (mobile) 
- Can essentially get an infinite score
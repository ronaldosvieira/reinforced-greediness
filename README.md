# unnamed-locm-bot
AI player for Legends of Code and Magic submitted to the [IEEE CEC 2020's Strategy Card Game AI Competition](https://jakubkowalski.tech/Projects/LOCM/CEC20). Made by Ronaldo Vieira, Luiz Chaimowicz and Anderson Tavares from Universidade Federal de Minas Gerais and Universidade Federal do Rio Grande do Sul.

## Installation
```python
pip install numpy scipy sortedcontainers
git clone https://github.com/ronaldosvieira/unnamed-locm-bot.git
cd unnamed-locm-bot
```

## Draft strategy
We use neural networks to choose cards. They are trained by reinforcement learning in a competitive self-play setting, and a separate network is used when playing as first player and second player. This is part of Ronaldo's master thesis.

## Playing strategy
We find the best combination of actions with a best-first search that considers only the current turn. A simplified version of the forward model available in the [gym-locm](https://github.com/ronaldosvieira/gym-locm) project is used. Our state evaluation function uses hand-tuned features kindly borrowed from the Fabbiamo bot, from the COG '19 edition of the SCGAI competition.

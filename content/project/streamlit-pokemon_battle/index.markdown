---
title: Predicting Pokemon Battles with Streamlit and DataRobot
author: Marshall Krassenstein
date: '2021-12-09'
slug: hi-hugo
categories: []
tags: [Python, Machine-Learning, Webapp]
subtitle: ''
summary: 'An I built that predicts the winner of pokemon battles'
authors: [Marshall Krassenstein]
lastmod: '2021-12-09'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---


DataRobot is good at Machine Learning. What it's not super good at is allowing users to build customizable webapps around their models. 
Meanwhile, my team at work didn't know too much about easy ways to code up small applications so they were a little bit stuck using our in house app builder. 
Enter Streamlit and the example app I whipped up to show why Pythonic webapps and machine learning go hand in hand. And yes, it's my second Pokemon app. Sue me if you don't like it..

![Battle Predicting in Action](battle_simulator_gif.gif)

Behold the Pokemon Battle Simulator (it doesn't actually simulate battles. It just queries a model to predict the end result based on Pokemon types and base stats)!
Trained using DataRobot, users can select a Pokemon or make one up and have it battle another Pokemon.

While simple, this application was great for my demo because it showed a lot of things DataRobot cannot do in house.

  1. Score user selected items determined by a dropdown.
  2. Utilize custom images (i.e. the winner of the Pokemon Battle).
  3. Better annotate what's happening.

There is obviously a ton more that one can do by leveraging Streamlit, but the app achieved its goal of demonstrating flexibility and ease of construction.

Play with the app below or view it [here](https://share.streamlit.io/mpkrass7/pokemon_battle_predictor/publish_to_streamlit/battle_simulator.py) (it's better). Don't forget to submit your favorite Pokemon!

<iframe src="https://share.streamlit.io/mpkrass7/pokemon_battle_predictor/publish_to_streamlit/battle_simulator.py" width="1152" height="900px"></iframe>




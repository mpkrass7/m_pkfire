---
title: Predicting Pokemon Battles with Streamlit and DataRobot
author: Marshall Krassenstein
date: '2021-12-09'
slug: battle-predictor
categories: []
tags: [python, machine-learning, web app]
subtitle: ''
summary: 'An app I built that predicts the winner of pokemon battles'
authors: [Marshall Krassenstein]
lastmod: '2021-12-09'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---


DataRobot is good at Machine Learning. What it's not super good at is allowing users to build customizable web apps around their models. 

Meanwhile, my team at work didn't know much about easy ways to code up small applications so they were a little bit stuck using our in house app builder. 

Enter streamlit and the example app I whipped up to show why pythonic web apps and machine learning go hand in hand. And yes, it's my second Pokemon app. Sue me if you don't like it..

![Battle Predicting in Action](battle_simulator_gif.gif)

Behold the Pokemon Battle Simulator (it doesn't actually simulate battles. It just queries a model to predict the end result based on Pokemon types and base stats)!
Trained using DataRobot, users can select a Pokemon or make one up and have it battle another Pokemon.

While simple, this application was great for my demo because it showed a lot of things DataRobot cannot do in house.

  1. Score user selected items determined by a dropdown.
  2. Utilize custom images (i.e. the winner of the Pokemon Battle).
  3. Better annotate what's happening.

There is obviously a ton more that one can do by leveraging streamlit, but the app achieved its goal of demonstrating flexibility and ease of construction.

View the app [here](https://pokepredict.streamlit.app/) and pin two Pokemon against each other in heated battle!

<!-- <iframe src="https://pokepredict.streamlit.app/" width="1152" height="900px"></iframe> -->




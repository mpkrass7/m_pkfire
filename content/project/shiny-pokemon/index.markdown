---
title: Two Way Analytics with Shiny and Postgres SQL
author: Marshall Krassenstein
date: '2021-09-18'
slug: pokemon-db
categories: []
tags: [R, shiny, web app, PCA]
subtitle: ''
summary: 'An app I built to show users Persistent Storage in Shiny.'
authors: [Marshall Krassenstein]
lastmod: '2020-06-18T13:33:09-04:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---


I spent much of my time at PNC Bank building web apps using R Shiny. One particular application that ate many hours of my time was called the 'WEASEL'. Effectively it was a Shiny app that served as a CRUD application for submitting project management workflow in Treasury Management. If it sounds boring, you're right operationally but the app building piece of it was pretty fun. This application unfortunately involved PII data and is thus lost to the public.

As a result of my foray into persistent data storage, however, I decided to make a different app that is alive and well for public consumption, and it is very much about Pokemon.

![Persistent Storage](poke_shiny1.gif)

As shown above, the application allows users to pick there favorite Pokemon among 6 generations, and store it in a Postgres SQL database. I love coming back to this app from time to time because every month or so, I see one or two new submissions. I don't know where people went to find this app but I find it certainly makes me happy ;).

In addition to the database piece I have a pretty pithy analysis page that uses Principal Component Analysis to put Pokemon on a two-dimensional plane for some directional comparison. Ever wonder how similar Psyduck is to Bulbasaur? Try looking at the tab marked 'analysis'. Getting the Pokemon images to dynamically render on my plotly object is still one of my favorite visualization accomplishments to date.

View the app below or follow the link [here](https://marshallp.shinyapps.io/2025PokemonAppShiny/). Don't forget to submit your favorite Pokemon!

<iframe src="https://marshallp.shinyapps.io/2025PokemonAppShiny/" width="1152" height="900px"></iframe>

#### Do you like Shiny and want persistent storage for free?

Follow the Medium article I wrote [here](https://medium.com/swlh/two-way-analytics-with-r-shiny-and-pokemon-e9eae225fd46)
Or look at my repo [here](https://github.com/mpkrass7/shiny_pokemon).

![PCA](poke_shiny2.gif)

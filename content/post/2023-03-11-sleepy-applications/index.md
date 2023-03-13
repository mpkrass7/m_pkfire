---
title: "Free hosting sites aren't allowed to make my apps go to sleep"
author: Marshall Krassenstein
date: '2023-03-11'
slug: sleepy-apps
categories: []
tags: [python, r, bash, cron, streamlit, shiny, dash, asyncio, selenium]
subtitle: ''
summary: 'A sneaky solution to keeping my web-apps awake 24/7'
authors: [Marshall Krassenstein]
featured: yes
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

I think anyone who has perused my website or ever worked with me before knows I'm a sucker for easy frameworks to make data oriented web applications. When I worked at PNC, I forced [R Shiny](https://shiny.rstudio.com/) to work for everything from CRUD applications to interactive networks graphs. At DataRobot I've built maybe a dozen applications using [Streamlit](https://streamlit.io/) and one or two more in [Dash](https://plotly.com/dash/) as well. Personally, I don't think it's incredibly gratifying to build an interactive application if it's just sitting on your computer. Otherwise, other people can't interact with it! So, when I finish building my stuff, I need to find a way to publish it. Doing this is increasingly easy. In fact, Streamlit and Shiny both offer a way to deploy publicly viewable web applications for free with [Streamlit Cloud](https://streamlit.io/cloud) and [shinyapps.io](https://www.shinyapps.io/).

Here's the problem though: Free services like these are great for getting started, but generally don't keep things  running 24/7. If an app doesn't get a view for a few days, it goes to sleep and the next person who comes to visit finds themselves looking at a message like this:

![Streamlit Sleeping App](images/sleepy_app.png)

Now your guest has to click a button and sit around for a server to spin up just so he can click around on your little widget. Not a great experience for anyone. To handle this, one could simply pay for a service like [Heroku](https://www.heroku.com/) or [AWS](https://aws.amazon.com/) to host their application without downtime. I'm kind of cheap though and I don't want to pay for something I can do for free if I can avoid it. Thus, I present a fix using the three players below.

### Key packages
- Selenium: Automates interacting with browsers
- Asyncio: Part of Python standard library for writing [concurrent](https://en.wikipedia.org/wiki/Concurrency_(computer_science)) code with async/await syntax
- Cron: Command line utility to trigger jobs on a schedule on Unix-like operating systems

### Workflow

After listing the tools I'm working with, it probably isn't too hard to guess at my plan for keeping my web applications up and running. I simply take a list of urls, open them, wait a couple of minutes and then close them in a headless browser with selenium. I then schedule it to run on my computer every day using `cron`. Ordinarily this script would take $(TimeWaiting * NumberOfApps)$ to finish (in this case 2 minutes * 7 urls). But because I've been doing a lot of asynchronous programming lately, I thought it would be cool to see how much faster I could make that finish in an elegant script.

```python

import asyncio
import time

from logzero import logger

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# List of free service hosted web applications I want to keep running
urls = [
    {
        "url": "https://pokepredict.streamlit.app/",
        "name": "Pokemon Battle Predictor",
    },
    {
        "url": "https://drplantclassifier.streamlit.app/",
        "name": "Plant Disease Classifier",
    },
    {
        "url": "https://amlbuddy.streamlit.app/",
        "name": "AML App",
    },
    {
        "url": "https://rollingstonalytics.streamlit.app/",
        "name": "Rolling Stone Top 500",
    },
    {
        "url": "https://statesmigrate.streamlit.app/",
        "name": "Migration App",
    },
    {
        "url": "https://utah-house-pricing.streamlit.app/",
        "name": "Utah Housing Market",
    },
    {
        "url": " https://marshallp.shinyapps.io/ShinyPokemonDB/",
        "name": "Shiny Pokemon DB",
    },
]


async def open_app(url, service, sleeptime=120):
    """Open a webpage, wait *sleeptime seconds, then close it"""

    # Configuration for headless browser (i.e. doesn't open a window)
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    logger.info(f"Opening {url['name']} at {url['url']}...")

    driver = webdriver.Chrome(service=Service(service), options=chrome_options)

    # Go to url
    driver.get(url["url"])
    
    # Wait for 2 minutes and let function execute for other apps in the mean time
    await asyncio.sleep(sleeptime)

    # Close the browser
    driver.close()
    logger.info(f"Closed {url['name']} at {url['url']}")
    return 200


async def main(urls):

    service = ChromeDriverManager().install()
    start = time.time()
    # Run everything concurrently
    await asyncio.gather(*(open_app(url, service) for url in urls))
    logger.info(f"Finished in {time.time() - start} seconds")

if __name__ == "__main__":
    asyncio.run(main(urls))
```

## Asynchronous programming and Selenium in action

Here's what the script looks like when I set it to wait on a website for 10 seconds. Notice how all of the applications are opened before the script beings closing them but then closes a bunch at once. That's because asynchronous functions allow other programs to execute while they wait for something (in this case a set period of time). 

![](images/run_refresh_script.gif)

The act of opening a webpage takes a couple of seconds which is why this script finished in 30 seconds as opposed to ~10 seconds but it's still faster than running them all in sequence. And if the wait is longer the effect, is much more dramatic. When I allow the function to execute with its default wait length of two minutes, it actually finishes 141 seconds, just 20 seconds longer than a single url was set to wait. In other words it only takes it about 17% to finish running on 7 urls than it takes a synchronous program to run on one url. In my real person job, we use asynchronous programming often when waiting rest API calls to post or return results.

## Scheduling the job

Once I wrote the Python code, the rest was easy. I created a bash small bash script that runs my Python function..

```bash
echo "Refreshing sleepy webapps.."
/Users/marshall.krassenstein/.pyenv/versions/general_env/bin/python run_sleepy_apps.py
```

and then I scheduled my bash script to trigger with a simple cron job

```
30 8,14 * * * cd /Users/marshall.krassenstein/desktop/random_projects/run_sleepy_apps && bash run_sleepy_refresh.sh > /tmp/throwaway.log 2>/tmp/webapp_refresh.log
```

Breaking this down:
`30 8,14 * * *`: Run this every day at 8:30am and 2:30pm.
`cd /Users/marshall.krassenstein/desktop/random_projects/run_sleepy_apps`: Change directory to where the script is located.
`bash run_sleepy_refresh.sh`: Run the bash script.
`> /tmp/throwaway.log 2>/tmp/webapp_refresh.log`: Redirect the output to a log file in case I want to check it later.

And just like that, we have a script that keeps our web applications up and running without having to pay for a server! Hurray!

### A Small Analysis on Time Savings

I honestly planned to stop writing there but I thought it would be fun to do a quick analysis of the amount of time running this script asynchronously can save when varying the amount of urls or time waited. So, I decided to run this script 100 times on a grid of 10 different numbers of requests (10-100) and 10 different wait times (10-100).
Below I compare the run time in seconds of synchronous run calls with the run time of asynchronous run calls. No surprise that we see the asynchronous version of the script run an order of magnitude faster by the time we get to 50 requests and 50 seconds of time to execute each request. What's more interesting is that my program in this case scaled far more when increasing the time taken for one call rather than the number of calls. I have a feeling this may have to do with the script opening up so many headless browsers and freezing my computer as the number of calls increased :). 
|                                   |                                    |
| :-------------------------------: | :--------------------------------: |
| ![](images/sync_time_contour.png) | ![](images/async_time_contour.png) |

A contour of the efficiency gains is shown below:
![](images/async_savings_contour.png)

## Enhancements I should make at some point:

- Cron jobs are great but a more robust scheduler that lives outside of my computer like Airflow would make this run even when my computer is off
- My script doesn't press anything on the screen to wake apps up if they're already sleeping. It would probably take two more lines in Selenium to check if a button to 'wake app up' exists on the browser.


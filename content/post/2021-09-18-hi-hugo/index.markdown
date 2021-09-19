---
title: Hi Hugo
author: R package build
date: '2020-09-18'
slug: hi-hugo
categories: []
tags: []
subtitle: ''
summary: 'A demonstration of putting a markdown file together'
authors: []
lastmod: '2021-09-18T13:33:09-04:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---


```r
# This is a comment
summary(Orange)
```

```
##  Tree       age         circumference  
##  3:7   Min.   : 118.0   Min.   : 30.0  
##  1:7   1st Qu.: 484.0   1st Qu.: 65.5  
##  5:7   Median :1004.0   Median :115.0  
##  2:7   Mean   : 922.1   Mean   :115.9  
##  4:7   3rd Qu.:1372.0   3rd Qu.:161.5  
##        Max.   :1582.0   Max.   :214.0
```


```r
library(ggplot2)
oplot <- ggplot(Orange, aes(x = age, 
                   y = circumference, 
                   colour = Tree)) +
  geom_point() +
  geom_line() +
  guides(colour = FALSE) +
  theme_bw()
oplot
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-1.png" width="672" />


## Overview/Background
During my work as a data scientist/consultant for Conservation International, the monitoring team
expressed interest in being able to automatically classify plants as native, alien, or invasive in 
certain landscapes.

Automating the process for classifying invasives proved a little more straightforward, since there
was a centralized database for invasive species. I developed a tool that combines webscraping and
API calls to databases that accomplishes this. I packaged this into a Shiny App, which you can 
see the source code for [here](https://github.com/johannesnelson/PPC_Data_Analysis_Pipeline/blob/main/Scripts/inv_app/app.R) 
or use directly on [Shiny](https://johannes-nelson.shinyapps.io/invasive_species_scanner/).

The challenge in doing something like this for 'nativeness' or 'alienness' arises from the fact that
there isn't a centralized database that tracks native plants all over the world. Some exists for certain
regions, but since the projects I was analyzing were based in dozens of different countries, I needed another
solution.

## The LLM-powered Classifier
My solution involved writing a script to process species-country pairings that essentially does the following:
1.) Query Wikipedia for information about the plant.
2.) Filter the page content that is returned so that only sentences with relevant key words are included
(i.e. native, alien, endemic, range, invasive, etc.)
3.) Using LangChain to engineer prompts and bring an LLM into the loop, parse this wikipedia context for specific, 
relevant information like...
* Native range
* Alien range
4.) Only using the information that you extracted, make a classifcation decision.

I talk about the process a bit more in depth here, and I put it all into a Streamlit app
with a runnable demo here. Aside from the demo, which you can run by clicking the 'run' demo
button at the top, you can also input your own text to process a single pairing or your own 
CSV with columns for 'species' and 'country' to process multiple. 

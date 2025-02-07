# app.R
library(shiny)
library(leaflet)
library(readr)
library(dplyr)
library(ggplot2)  # Loaded in case you want to explore the ggplot output separately

ui <- fluidPage(
  titlePanel("Weather and EMR Map"),
  # Display the interactive leaflet map
  leafletOutput("map", height = "600px")
)

server <- function(input, output, session) {
  
  # Set the raw data directory
  dir.raw <- './data/'
  
  ## Weather Data
  weather_file <- paste0(dir.raw, 'weather_station_metadata.rds')
  

  # Read the actual data (skipping the header)
  df <- readRDS(weather_file)
  
  # (Optional) Create a ggplot of station locations
  # p <- ggplot(df, aes(x = station_longitude, y = station_latitude)) +
  #   geom_point() +
  #   theme_bw()
  # print(p)
  
  ## EMR Data
  emr_file <- paste0(dir.raw, 'EMR_address_sample.rds')

  df.emr.geo <- readRDS(emr_file)
  
  ## Create the leaflet map
  output$map <- renderLeaflet({
    leaflet() %>%
      addTiles() %>%
      addProviderTiles(providers$OpenStreetMap) %>%
      addCircleMarkers(
        data = df, 
        lng = ~station_longitude, 
        lat = ~station_latitude,
        color = "red",
        popup = ~paste(
          "<strong> station_name: </strong>", station_name, "<br>",
          "<strong> station_elevation: </strong>", station_elevation, "<br>",
          "<strong> first_year: </strong>", first_year, "<br>",
          "<strong> last_year: </strong>", last_year, "<br>"
        ),
        stroke = FALSE, fillOpacity = 0.5
      ) %>%
      addCircleMarkers(
        data = df.emr.geo,
        lng = ~longitude, 
        lat = ~latitude,
        popup = ~adminstrative_area,
        color = "blue", 
        radius = 2,
        stroke = FALSE, fillOpacity = 0.5
      ) %>%
      setView(lng = -0.119, lat = 51.525, zoom = 10)
  })
}

shinyApp(ui = ui, server = server)

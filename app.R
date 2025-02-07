# app.R
library(shiny)
library(leaflet)
library(markdown)
library(readr)
library(dplyr)
library(ggplot2)  # Loaded in case you want to explore the ggplot output separately


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

# Compute ranges for first_year and last_year from the weather data
first_year_range <- range(df$first_year, na.rm = TRUE)
last_year_range  <- range(df$last_year, na.rm = TRUE)
  
#--- UI ---
ui <- fluidPage(
  titlePanel("Urban Nature, Heatwave, and Health"),
  
  # Place the project description immediately after the title panel
  includeMarkdown("description.md"),
  
  sidebarLayout(
    sidebarPanel(
      sliderInput("first_year_filter", "Filter by First Year",
                  min = first_year_range[1],
                  max = first_year_range[2],
                  value = first_year_range,
                  step = 1),
      sliderInput("last_year_filter", "Filter by Last Year",
                  min = last_year_range[1],
                  max = last_year_range[2],
                  value = last_year_range,
                  step = 1)
    ),
    mainPanel(
      # # Include Markdown content for the project description
      # includeMarkdown("description.md"),
      leafletOutput("map", height = "900px")
    )
  )
)

#--- Server ---
server <- function(input, output, session) {
  
  # Create a reactive expression that filters the weather data based on user input
  filtered_weather <- reactive({
    df %>% 
      filter(first_year >= input$first_year_filter[1],
             first_year <= input$first_year_filter[2],
             last_year  >= input$last_year_filter[1],
             last_year  <= input$last_year_filter[2])
  })
  
  # Render the leaflet map
  output$map <- renderLeaflet({
    leaflet() %>%
      addTiles() %>%
      addProviderTiles(providers$OpenStreetMap) %>%
      # Add markers for weather stations based on filtered data
      addCircleMarkers(
        data = filtered_weather(),
        lng = ~station_longitude,
        lat = ~station_latitude,
        color = "red",
        popup = ~paste(
          "<strong>Station Name:</strong>", station_name, "<br>",
          "<strong>Elevation:</strong>", station_elevation, "<br>",
          "<strong>First Year:</strong>", first_year, "<br>",
          "<strong>Last Year:</strong>", last_year, "<br>"
        ),
        stroke = FALSE, fillOpacity = 0.5
      ) %>%
      # Add markers for EMR data
      addCircleMarkers(
        data = df.emr.geo,
        lng = ~longitude,
        lat = ~latitude,
        popup = ~adminstrative_area,
        color = "blue",
        radius = 3,
        stroke = FALSE, fillOpacity = 0.5
      ) %>%
      setView(lng = -0.119, lat = 51.4, zoom = 11)
  })
  
  # Optional: Use an observer with leafletProxy to update markers without re-rendering the whole map.
  # observe({
  #   leafletProxy("map", data = filtered_weather()) %>%
  #     clearMarkers() %>%
  #     addCircleMarkers(
  #       lng = ~station_longitude,
  #       lat = ~station_latitude,
  #       color = "red",
  #       popup = ~paste(
  #         "<strong>Station Name:</strong>", station_name, "<br>",
  #         "<strong>Elevation:</strong>", station_elevation, "<br>",
  #         "<strong>First Year:</strong>", first_year, "<br>",
  #         "<strong>Last Year:</strong>", last_year, "<br>"
  #       ),
  #       stroke = FALSE, fillOpacity = 0.5
  #     )
  # })
}

# Run the application 
shinyApp(ui = ui, server = server)

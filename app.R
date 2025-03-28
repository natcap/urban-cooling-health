# app.R
library(shiny)
library(leaflet)
library(markdown)
library(readr)
library(dplyr)
library(ggplot2)  # Loaded in case you want to explore the ggplot output separately
library(plotly) 


# Set the raw data directory
dir.raw <- './data/'

## Weather Data --------------------------------------------------------------------------

### 1.1 station info ---------
weather_file <- paste0(dir.raw, 'weather_station_metadata.rds')
df <- readRDS(weather_file)

# (Optional) Create a ggplot of station locations
# p <- ggplot(df, aes(x = station_longitude, y = station_latitude)) +
#   geom_point() +
#   theme_bw()
# print(p)


### 1.2 temperature data -----
dat_stat <- readRDS(file = paste0(dir.raw, 'london_weather_obs_stat.RDS'))



## EMR Data ------------------------------------------------------------------------------
emr_file <- paste0(dir.raw, 'EMR_address_sample.rds')

df.emr.geo <- readRDS(emr_file)

# Compute ranges for first_year and last_year from the weather data
first_year_range <- range(df$first_year, na.rm = TRUE)
last_year_range  <- range(df$last_year, na.rm = TRUE)
  


## trigger the filtering and update the plot ---------------------------------------------
# Reactive values to store selected src_id
selected_src_id <- reactiveVal(NULL)



#--- UI --- ------------------------------------------------------------------------------
ui <- fluidPage(
  titlePanel("Urban Nature, Heatwave, and Health"),
  
  # Place the project description immediately after the title panel
  includeMarkdown("description.md"),
  
  fluidRow(
    column(12,
           sidebarPanel(
             sliderInput("first_year_filter", "Filter by First Year",
                         min = min(df$first_year, na.rm = TRUE),
                         max = max(df$first_year, na.rm = TRUE),
                         value = range(df$first_year, na.rm = TRUE),
                         step = 1),
             sliderInput("last_year_filter", "Filter by Last Year",
                         min = min(df$last_year, na.rm = TRUE),
                         max = max(df$last_year, na.rm = TRUE),
                         value = range(df$last_year, na.rm = TRUE),
                         step = 1)
           )
    )
  ),
  
  fluidRow(
    column(6, leafletOutput("map", height = "600px")),
    column(6, 
           tabsetPanel(
             tabPanel("Air Temperature", plotlyOutput("stat_plot", height = "600px")),
             tabPanel("Relative Humidity", plotlyOutput("humidity_plot", height = "600px"))
           )
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
        layerId = ~src_id,
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
  
  # Update src_id when a point is clicked ------------------------------------------------
  observeEvent(input$map_marker_click, {
    selected_src_id(input$map_marker_click$id)
  })
  
  # Filter data based on selected src_id
  filtered_data <- reactive({
    req(selected_src_id())  # Ensure src_id is selected
    dat_stat %>%
      filter(src_id == selected_src_id())
  })
  
  
  # # 3.1. Plot static data by month
  # output$stat_plot <- renderPlot({
  #   req(filtered_data())
  #   ggplot(filtered_data(), aes(x = date, y = air_temperature, color = day_night, label = round(air_temperature, 1))) +
  #     geom_point(alpha = 0.5) +
  #     geom_line(alpha = 0.5) +
  #     geom_text(check_overlap = TRUE, vjust = -0.5, hjust = 0.5, show.legend = F) +
  #     theme_minimal() +
  #     xlab('Date')
  # })
  # 
  
  # 3.2. Plot interactive data by month --
  output$stat_plot <- renderPlotly({
    req(filtered_data())
    df_plot <- filtered_data() %>% # Assign data for debugging
      dplyr::mutate(tooltip = paste0("Date: ", format(date, "%Y-%m"), "\nTemp: ", round(air_temperature, 1), "°C")) 
    
    # Avoid plotting if no data
    if (nrow(df_plot) == 0) {return(NULL)}
    
    p <- ggplot(df_plot, aes(x = date, y = air_temperature, color = day_night, text = tooltip)) +
      geom_point(alpha = 0.5) +
      geom_line(alpha = 0.5) +
      theme_minimal() +
      xlab('Date')
    ggplotly(p, tooltip = "text")  # Convert ggplot to interactive plotly
  })
  
  output$humidity_plot <- renderPlotly({
    req(filtered_data())
    df_plot <- filtered_data() %>%
     dplyr::mutate(tooltip = paste0("Date: ", format(date, "%Y-%m"), "\nRH: ", round(rltv_hum, 1), "%"))
    
    if (nrow(df_plot) == 0) return(NULL)
    
    p <- ggplot(df_plot, aes(x = date, y = rltv_hum, color = day_night, text = tooltip)) +
      geom_point(alpha = 0.5) +
      geom_line(alpha = 0.5) +
      theme_minimal() +
      xlab('Date') +
      ylab('Relative humidity (%)')
    
    ggplotly(p, tooltip = "text")
  })
  
  
}

# Run the application 
shinyApp(ui = ui, server = server)

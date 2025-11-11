library(tidyverse)
library(car)
library(broom)
library(effectsize)
library(interactions)

spotify <- read_csv("merged_billboard_spotify_matched_only.csv", show_col_types = FALSE)

#Clean Data
spotify_clean <- spotify %>%
  select(`weeks-on-board`, genre, energy, danceability, valence, loudness) %>%
  rename(longevity = `weeks-on-board`) %>%
  drop_na()

#Simplify the genres
top_genres <- spotify_clean %>%
  count(genre) %>%
  arrange(desc(n)) %>%
  slice_head(n = 6) %>%
  pull(genre)

spotify_clean <- spotify_clean %>%
  mutate(genre = ifelse(genre %in% top_genres, genre, "Other"),
         genre = as.factor(genre))

# Standardize numeric predictors
spotify_clean <- spotify_clean %>%
  mutate(across(c(energy, danceability, valence, loudness),
                ~scale(.x)[,1], .names = "z_{col}"))

#Visualizations
ggplot(spotify_clean, aes(x = longevity)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Song Longevity", x = "Weeks on Billboard", y = "Count")

ggplot(spotify_clean, aes(x = genre, y = longevity)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Song Longevity by Genre", x = "Genre", y = "Weeks on Billboard") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Regression with interaction terms
model_full <- lm(longevity ~ genre * (z_energy + z_danceability + z_valence + z_loudness),
                 data = spotify_clean)

# Reduced model without interactions
model_reduced <- lm(longevity ~ genre + z_energy + z_danceability + z_valence + z_loudness,
                    data = spotify_clean)

# Compare models
anova(model_reduced, model_full)

# Type III ANOVA
options(contrasts = c("contr.sum", "contr.poly"))
Anova(model_full, type = "III")

# Effect size
eta_squared(model_full, partial = TRUE)

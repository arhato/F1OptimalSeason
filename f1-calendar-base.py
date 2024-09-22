import unittest
import math
import csv
import random
from simanneal import Annealer
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# constants that define the likelihood of two individuals having crossover
# performed and the probability that a child will be mutated. needed for the
# DEAP library
CXPB = 0.5
MUTPB = 0.2


# the unit tests to check that the simulation has been implemented correctly
class UnitTests(unittest.TestCase):
    # this will read in the track locations file and will pick out 5 fields to see if the file has been read correctly
    def testReadCSV(self):
        # read in the locations file
        rows = readCSVFile('track-locations.csv')

        # test that the corners and a middle value are read in correctly
        self.assertEqual('circuit', rows[0][0])
        self.assertEqual('Dec Temp', rows[0][14])
        self.assertEqual('Yas Marina', rows[22][0])
        self.assertEqual('26', rows[22][14])
        self.assertEqual('27', rows[11][8])

    # this will test to see if the column conversion works. here we will convert the latitude column and will test 5 values
    # as we are dealing with floating point we will use almost equals rather than a direct equality
    def testColToFloat(self):
        # read in the locations file and convert the latitude column to floats
        rows = readCSVFile('track-locations.csv')
        convertColToFloat(rows, 1)

        # check that 5 of the values have converted correctly
        self.assertAlmostEqual(26.0325, rows[1][1], delta=0.0001)
        self.assertAlmostEqual(24.4672, rows[22][1], delta=0.0001)
        self.assertAlmostEqual(40.3725, rows[4][1], delta=0.0001)
        self.assertAlmostEqual(30.1327, rows[18][1], delta=0.0001)
        self.assertAlmostEqual(25.49, rows[17][1], delta=0.0001)

    # this will test to see if the column conversion to int works. here we will convert one of the temperature columns and will
    # test 5 values to see that it worked correctly
    def testColToInt(self):
        # read in the locations file and convert the first of the temperature columns to ints
        rows = readCSVFile('track-locations.csv')
        convertColToInt(rows, 3)

        # check that the values are converted correctly
        self.assertEqual(20, rows[1][3])
        self.assertEqual(24, rows[22][3])
        self.assertEqual(4, rows[11][3])
        self.assertEqual(9, rows[16][3])
        self.assertEqual(23, rows[5][3])

    # this will test to see if the file conversion overall is successful for the track locations
    # it will read in the file and will test a string, float, and int from 2 rows to verify it worked correctly
    def testReadTrackLocations(self):
        # read in the locations file
        rows = readTrackLocations()

        # check the name, latitude, and final temp of the first race
        self.assertEqual(rows[0][0], 'Bahrain International Circuit')
        self.assertEqual(rows[0][14], 22)
        self.assertAlmostEqual(rows[0][1], 26.0325, delta=0.0001)

        # check the name, longitude, and initial temp of the last race
        self.assertEqual(rows[21][0], 'Yas Marina')
        self.assertEqual(rows[21][3], 24)
        self.assertAlmostEqual(rows[21][2], 54.603056, delta=0.0001)

    # tests to see if the race weekends file is read in correctly
    def testReadRaceWeekends(self):
        # read in the race weekends file
        weekends = readRaceWeekends()

        # check that bahrain is weekend 9 and abu dhabi is weekend 47
        self.assertEqual(weekends[0], 9)
        self.assertEqual(weekends[21], 47)

        # check that hungaroring is weekend 29
        self.assertEqual(weekends[10], 29)

    # tests to see if the sundays file is read in correctly
    def testReadSundays(self):
        # read in the sundays file and get the map of sundays back
        sundays = readSundays()

        # check to see the first sunday is january and the last sunday is december
        self.assertEqual(sundays[0], 0)
        self.assertEqual(sundays[51], 11)

        # check a few other random sundays
        self.assertEqual(sundays[10], 2)
        self.assertEqual(sundays[20], 4)
        self.assertEqual(sundays[30], 6)
        self.assertEqual(sundays[40], 9)

    # this will test to see if the haversine function will work correctly we will test 4 sets of locations
    def testHaversine(self):
        # read in the locations file with conversion
        rows = readTrackLocations()

        # check the distance of Bahrain against itself this should be zero
        self.assertAlmostEqual(haversine(rows, 0, 0), 0.0, delta=0.01)

        # check the distance of Bahrain against Silverstone this should be 5158.08 km
        self.assertAlmostEqual(haversine(rows, 0, 9), 5158.08, delta=0.01)

        # check the distance of silverstone against monza this should be 1039.49 Km
        self.assertAlmostEqual(haversine(rows, 13, 9), 1039.49, delta=0.01)

        # check the distance of monza to the red bull ring this should be 455.69 Km
        self.assertAlmostEqual(haversine(rows, 13, 8), 455.69, delta=0.01)

    # will test to see if the season distance calculation is correct using the 2023 calendar
    def testDistanceCalculation(self):
        # read in the locations & race weekends, generate the weekends, and calculate the season distance
        tracks = readTrackLocations()
        weekends = readRaceWeekends()

        # calculate the season distance using silverstone as the home track as this will be the case for 8 of the teams we will use monza
        # for the other two teams.
        self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 9), 185874.8866, delta=0.0001)
        self.assertAlmostEqual(calculateSeasonDistance(tracks, weekends, 13), 179336.2663, delta=0.0001)

    # will test that the temperature constraint is working this should fail as azerbijan should fail the test
    def testTempConstraint(self):
        # load in the tracks, race weekends, and the sundays
        tracks = readTrackLocations()
        weekends1 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 35, 37, 38, 40, 42, 43, 44, 46, 47]
        weekends2 = [9, 11, 43, 30, 37, 21, 40, 34, 22, 35, 29, 26, 27, 24, 44, 42, 46, 18, 38, 13, 17, 47]
        sundays = readSundays()

        # the test with the default calendar should be false because of azerbaijan
        self.assertEqual(checkTemperatureConstraint(tracks, weekends1, sundays), False)
        self.assertEqual(checkTemperatureConstraint(tracks, weekends2, sundays), True)

    # will test that we can detect four race weekends in a row.
    def testFourRaceInRow(self):
        # weekend patterns the first does not have four in a row the second does
        weekends1 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 35, 37, 38, 40, 42, 43, 44, 46, 47]
        weekends2 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 35, 37, 38, 41, 42, 43, 44, 46, 47]

        # the first should pass and the second should fail
        self.assertEqual(checkFourRaceInRow(weekends1), False)
        self.assertEqual(checkFourRaceInRow(weekends2), True)

    # will test that we can detect a period for a summer shutdown in july and/or august
    def testSummerShutdown(self):
        # weekend patterns the first has a summer shutdown the second doesn't
        weekends1 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 35, 37, 38, 40, 42, 43, 44, 46, 47]
        weekends2 = [9, 11, 13, 17, 18, 21, 22, 24, 26, 28, 30, 32, 34, 35, 37, 38, 40, 42, 43, 44, 46, 47]

        # the first should pass and the second should fail
        self.assertEqual(checkSummerShutdown(weekends1), True)
        self.assertEqual(checkSummerShutdown(weekends2), False)


# function that will calculate the total distance for the season assuming a given racetrack as the home racetrack
# the following will be assumed:
# - on a weekend where there is no race the team will return home
# - on a weekend in a double or triple header a team will travel straight to the next race and won't go back home
# - the preseason test will always take place in Bahrain
# - for the summer shutdown and off season the team will return home
def calculateSeasonDistance(tracks, weekends, home):
    current_location = home
    season_distance = 0
    for i in range(1, 52 + 1):
        if i in weekends:
            season_distance += haversine(tracks, current_location, weekends.index(i))
            current_location = weekends.index(i)
        else:
            season_distance += haversine(tracks, current_location, home)
            current_location = home
    return season_distance


# function that will check to see if there is anywhere in our weekends where four races appear in a row. True indicates that we have four in a row
def checkFourRaceInRow(weekends):
    checker = False
    counter = 1
    for i in range(1, len(weekends)):
        if weekends[i] == weekends[i - 1] + 1:
            counter += 1
            if counter >= 3:
                checker = True
                break
        else:
            counter = 0
    return checker


# function that will check to see if the temperature constraint for all races is satisfied. The temperature
# constraint is that a minimum temperature of 20 degrees for the month is required for a race to run
def checkTemperatureConstraint(tracks, weekends, sundays):
    checker = True
    for i in range(len(weekends)):
        month = sundays[weekends[i]] + 1
        temp = tracks[i][month + 2]
        if 20 <= temp >= 35:
            checker = False
            break
    return checker


# function that will check to see if there is a four week gap anywhere in july and august. we will need this for the summer shutdown.
# the way this is defined is that we have a gap of three weekends between successive races.
def checkSummerShutdown(weekends):
    checker = False
    for i in range(len(weekends)):
        if 26 < weekends[i] < 36:
            if weekends[i] - weekends[i - 1] > 3:
                checker = True
                break
    return checker


# function that will take in the set of rows and will convert the given column index into floating point values
# this assumes the header in the CSV file is still present, so it will skip the first row
def convertColToFloat(rows, column_index):
    for row in rows[1:]:
        row[column_index] = float(row[column_index])
    return rows


# funciton that will take in a set of rows and will convert the given column index into integer values
# this assumes the header in the CSV file is still present, so it will skip the first row
def convertColToInt(rows, column_index):
    for row in rows[1:]:
        row[column_index] = int(row[column_index])
    return rows


# function that will use the haversine formula to calculate the distance in Km given two latitude/longitude pairs
# it will take in an index to two rows, and extract the latitude and longitude before the calculation.
def haversine(rows, location1, location2):
    # distance between latitudes
    # and longitudes

    lat1 = rows[location1][1]
    lon1 = rows[location1][2]
    lat2 = rows[location2][1]
    lon2 = rows[location2][2]

    # formula from geeksforgeeks.org
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0

    # apply formulae
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
         math.cos(lat1) * math.cos(lat2));
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    return rad * c


# prints out the itinerary that was generated on a weekend by weekend basis starting from the preaseason test
def printItinerary(tracks, weekends, home, sundays):
    current_location = home
    home_track = tracks[home][0]
    print(f"Home: {home_track}")

    for i in range(1, 52 + 1):
        if i in weekends:
            next_location = weekends.index(i)
            next_track = tracks[next_location][0]
            month = sundays[i] + 1
            next_race_temp = tracks[next_location][month + 2]

            if current_location == home:
                print(
                    f"Week: {i}; Travelling from Home to {next_track}. Race temperature is expected to be {next_race_temp} degrees")
            else:
                current_track = tracks[current_location][0]
                print(
                    f"Week: {i}; Travelling directly from {current_track} to {next_track}. Race temperature is expected to be {next_race_temp} degrees")

            current_location = next_location
        else:
            if current_location == home:
                print(f"Week: {i}; Staying at Home.")
            else:
                current_track = tracks[current_location][0]
                print(f"Week: {i}; Travelling home from {current_track}")
                current_location = home  # Stay home until the next race weekend


# function that will take in the given CSV file and will read in its entire contents
# and return a list of lists
def readCSVFile(file):
    # the rows to return
    rows = []
    rows = []

    # open the file for reading and give it to the CSV reader
    csv_file = open(file)
    csv_reader = csv.reader(csv_file, delimiter=',')

    # read in each row and append it to the list of rows.
    for row in csv_reader:
        rows.append(row)

    # close the file when reading is finished
    csv_file.close()

    # return the rows at the end of the function
    return rows


# function that will read in the race weekends file and will perform all necessary conversions on it
def readRaceWeekends():
    rows = readCSVFile('race-weekends.csv')
    convertColToInt(rows, 1)
    rows = rows[1:]
    new_rows = []
    for row in rows:
        temp = [(row[1])]
        new_rows.append(temp)
    result = []
    for row in new_rows:
        result.append(row[0])
    return result


# function that will read in the sundays file that will map the sundays to a list. each sunday maps to a month. we will need this for temperature comparisons later on
def readSundays():
    rows = readCSVFile('sundays.csv')
    rows = rows[1:]
    new_rows = []
    for row in rows:
        temp = [int(row[1])]
        new_rows.append(temp)
    result = []
    for row in new_rows:
        result.append(row[0])
    return result


# function that will read the track locations file and will perform all necessary conversions on it
def readTrackLocations():
    rows = readCSVFile('track-locations.csv')
    rows = rows[1:]
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            if j == 0:
                rows[i][j] = str(rows[i][j])
            elif j == 1 or j == 2:
                rows[i][j] = float(rows[i][j])
            else:
                rows[i][j] = int(rows[i][j])
    return rows


class RacingSchedule(Annealer):
    def __init__(self, state, tracks, weekends, home_track, sundays):
        super(RacingSchedule, self).__init__(state)
        self.tracks = tracks
        self.weekends = weekends
        self.home_track = home_track
        self.sundays = sundays

    def move(self):
        fixed_races = [9, 21, 47]  # Weekends for Bahrain, Monaco, and Abu Dhabi
        # pick two random races
        race1, race2 = random.sample([w for w in self.weekends if w not in fixed_races], 2)
        # get their indices
        index1, index2 = self.state.index(race1), self.state.index(race2)
        # swap them
        self.state[index1], self.state[index2] = self.state[index2], self.state[index1]
        # check to see if the temperature constraint is satisfied
        if not checkTemperatureConstraint(self.tracks, self.state, self.sundays):
            # if it isn't then swap them back
            self.state[index1], self.state[index2] = self.state[index2], self.state[index1]
        else:
            # calculate the season distance
            self.energy()

    def energy(self):
        # calculate the season distance
        return calculateSeasonDistance(self.tracks, self.state, self.home_track)


class RacingScheduleFree(Annealer):
    def __init__(self, state, tracks, weekends, home_track, sundays):
        super(RacingScheduleFree, self).__init__(state)
        self.tracks = tracks
        self.weekends = weekends
        self.home_track = home_track
        self.sundays = sundays

    def move(self):
        # Randomly choose between two move types
        move_type = random.choice(["swap", "move"])
        fixed_races = [9, 21, 47]  # Weekends for Bahrain, Monaco, and Abu Dhabi

        if move_type == "swap":
            # pick two random races
            race1, race2 = random.sample([w for w in self.state if w not in fixed_races], 2)
            # get their indices
            index1, index2 = self.state.index(race1), self.state.index(race2)
            # swap them
            self.state[index1], self.state[index2] = self.state[index2], self.state[index1]
        else:
            # select a random race
            race_to_move = random.choice(self.state)
            # get its index
            index = self.state.index(race_to_move)
            # randomly choose a direction to move it
            move_direction = random.choice([-1, 1])  # -1 for backward, 1 for forward
            # calculate the target week
            target_week = self.state[index] + move_direction
            # check to see if the target week is valid
            if 1 <= target_week <= 52 and target_week not in fixed_races and target_week not in self.state:
                self.state[index] = target_week

        # Check constraints
        if (
                checkTemperatureConstraint(self.tracks, self.state, self.sundays) and
                not checkFourRaceInRow(self.state) and
                len(set(self.state)) == len(self.state) == 22 and
                checkSummerShutdown(self.state) and
                self.state[0] == 9 and
                self.state[-1] == 47 and
                self.state[10] == 21
        ):
            self.energy()
        else:
            # If constraints are violated, revert the move
            if move_type == "swap":
                # Swap back
                self.state[index1], self.state[index2] = self.state[index2], self.state[index1]
            else:
                # Move back
                self.state[index] = self.state[index] - move_direction

    def energy(self):
        # calculate the season distance
        return calculateSeasonDistance(self.tracks, self.state, self.home_track)


def simulated_annealing(tracks, weekends, home_track, sundays):
    initial_state = weekends.copy()
    problem = RacingSchedule(initial_state, tracks, weekends, home_track, sundays)
    problem.Tmax = 100.0  # Initial temperature
    problem.Tmin = 1e-3  # Final temperature
    problem.steps = 200000
    best_schedule, energy = problem.anneal()

    return best_schedule, energy


def free_simulated_annealing(tracks, weekends, home_track, sundays):
    initial_state2 = weekends.copy()
    problem = RacingScheduleFree(initial_state2, tracks, weekends, home_track, sundays)
    problem.Tmax = 100.0  # Initial temperature
    problem.Tmin = 1e-3  # Final temperature
    problem.steps = 400000
    best_schedule, energy = problem.anneal()

    return best_schedule, energy


# function that will run the simulated annealing case for shortening the distance seperately for both silverstone and monza. it will also do a free calendar experiement
# to see if it can be cut down further
def SAcases(tracks, weekends, sundays):
    # Case 1: Best suited to teams with Silverstone as their home track
    best_silverstone_schedule, best_silverstone_energy = simulated_annealing(tracks, weekends, 9, sundays)

    # Case 2: Best suited to teams with Monza as their home track
    best_monza_schedule, best_monza_energy = simulated_annealing(tracks, weekends, 13, sundays)

    # Case 3: Best suited to teams with Silverstone as their home track, but with a free calendar
    best_free_calendar_schedule, best_free_calendar_energy = free_simulated_annealing(tracks, weekends, 9, sundays)

    print("-----------------------------------------------------------------------------------")
    print("Simulated Annealing:")
    print("Best schedule for teams with Silverstone as home track:", best_silverstone_schedule)
    print("Best energy for teams with Silverstone as home track:", best_silverstone_energy)
    print("Best schedule for teams with Monza as home track:", best_monza_schedule)
    print("Best energy for teams with Monza as home track:", best_monza_energy)
    print("Best schedule for a free calendar with Silverstone as home track:", best_free_calendar_schedule)
    print("Best energy for a free calendar with Silverstone as home track:", best_free_calendar_energy)


# Function to initialize a population of individuals
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        # a random individual (calendar) respecting constraints
        individual = generate_random_individual()
        population.append(individual)
    return population


def generate_random_individual():
    # Generate a random individual (calendar) respecting constraints
    random_weekends = readRaceWeekends()
    shuffled_weekends = random_weekends.copy()
    # Shuffle the weekends
    random.shuffle(shuffled_weekends)
    # Keep shuffling until we get a valid calendar
    while (
            not checkTemperatureConstraint(tracks, shuffled_weekends, sundays) or
            checkFourRaceInRow(shuffled_weekends) or
            len(set(shuffled_weekends)) != len(shuffled_weekends) or
            not checkSummerShutdown(shuffled_weekends) or
            shuffled_weekends[0] != 9 or
            shuffled_weekends[-1] != 47 or
            shuffled_weekends[10] != 21
    ):
        # Shuffle the weekends
        random.shuffle(shuffled_weekends)

    return shuffled_weekends


# Function to evaluate the fitness of an individual
def calculate_fitness(individual):
    # Implement logic to calculate the fitness of an individual
    # Consider factors such as total distance, adherence to constraints, etc.
    fitness_score = calculateSeasonDistance(tracks, individual, 9)
    return fitness_score


# Function to perform roulette wheel selection
def roulette_wheel_selection(population, fitness_scores):
    # Add a small constant to avoid zero total weights
    fitness_scores_with_constant = [score + 1e-10 for score in fitness_scores]

    # Select individuals based on their fitness using roulette wheel selection
    selected_indices = random.choices(range(len(population)), weights=fitness_scores_with_constant, k=len(population))
    selected_population = [population[i] for i in selected_indices]
    return selected_population


# Function to perform crossover (recombination)
def crossover(parent1, parent2):
    # Calculate fitness scores for the parents
    fitness_parent1 = calculate_fitness(parent1)
    fitness_parent2 = calculate_fitness(parent2)

    # Calculate probabilities based on fitness scores
    total_fitness = fitness_parent1 + fitness_parent2
    prob_parent1 = fitness_parent1 / total_fitness

    # Perform roulette wheel selection for crossover
    if random.random() < prob_parent1:
        # Select parent1 as the primary parent
        primary_parent = parent1
        secondary_parent = parent2
    else:
        # Select parent2 as the primary parent
        primary_parent = parent2
        secondary_parent = parent1

    # Perform crossover to create offspring
    crossover_point = random.randint(0, len(parent1) - 1)
    offspring1 = primary_parent[:crossover_point] + secondary_parent[crossover_point:]
    offspring2 = secondary_parent[:crossover_point] + primary_parent[crossover_point:]

    return offspring1, offspring2


# Function to perform mutation
def mutate(individual):
    fixed_races = [9, 21, 47]  # Weekends for Bahrain, Monaco, and Abu Dhabi
    mutated_individual = individual.copy()
    race_to_move = random.choice(mutated_individual)
    # get its index
    index = mutated_individual.index(race_to_move)
    # randomly choose a direction to move it
    move_direction = random.choice([-1, 1])  # -1 for backward, 1 for forward
    # calculate the target week
    target_week = mutated_individual[index] + move_direction
    # check to see if the target week is valid
    if 1 <= target_week <= 52 and target_week not in fixed_races and target_week not in mutated_individual:
        mutated_individual[index] = target_week
    # return the mutated individual
    return mutated_individual


# Function to execute the genetic algorithm
def genetic_algorithm(population_size, generations):
    # Initialize population
    population = initialize_population(population_size)

    for generation in range(generations):
        # Evaluate fitness of each individual in the population
        fitness_scores = [calculate_fitness(individual) for individual in population]

        # Select individuals for the next generation using roulette wheel selection
        selected_population = roulette_wheel_selection(population, fitness_scores)

        # Create the next generation through crossover
        next_generation = []
        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            offspring = crossover(parent1, parent2)
            next_generation.extend([mutate(offspring[0]), mutate(offspring[1])])

        # Replace the current population with the new generation
        population = next_generation

    # Identify the best individual in the final generation
    best_individual = max(population, key=calculate_fitness)

    return best_individual


def GAcases():
    best_calendar = genetic_algorithm(population_size=300, generations=1000)
    print("-----------------------------------------------------------------------------------")
    print("Genetic Algorithm:")
    print("Best Calendar:", best_calendar)


if __name__ == '__main__':
    unittest.main(exit=False)
    tracks = readTrackLocations()
    weekends = readRaceWeekends()
    sundays = readSundays()
    print("_____________________________________________________________________________________")
    print("F1 Racing Calendar For Silverstone as Home Track")
    printItinerary(tracks, weekends, 9, sundays)
    print("_____________________________________________________________________________________")

    # run the cases for simulated annealing
    SAcases(tracks, weekends, sundays)
    # run the cases for the genetic algorithm
    GAcases()
    
    

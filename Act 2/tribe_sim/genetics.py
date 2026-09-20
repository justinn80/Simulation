import random
import math
from entities import Gatherer
from config import *

class GeneticAlgorithm:
    def __init__(self):
        self.generation = 1
        self.fitness_history = []
        self.trait_history = []  # Track trait averages over generations
    
    def create_initial_population(self):
        population = []
        for _ in range(INITIAL_POPULATION):
            gatherer = Gatherer()
            population.append(gatherer)
        return population
    
    def evaluate_fitness(self, population):
        fitness_scores = []
        for gatherer in population:
            fitness = gatherer.calculate_fitness()
            fitness_scores.append((gatherer, fitness))
        
        # Sort by fitness (highest first)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        return fitness_scores
    
    def select_survivors(self, fitness_scores):
        # Minimal version: just take top 50% of population
        amount_alive = max(1, len(fitness_scores) // 2)  # Top 50%
        survivors = [fitness_scores[0][0]]
        while len(survivors) < amount_alive:
            remaining_life = min(3,len(fitness_scores))
            tournament = random.sample(fitness_scores, reamining_life)
            champion = max(tournament, key=lambda x: x[1])[0]
            if champion not in survivors:
                survivors.append(champion)
        return survivors
    
    def crossover(self, parent1, parent2):
        child_genes = {}
        for gene_name in parent1.genes:
            # 50% chance to inherit from each parent
            if random.random() < 0.5:
                child_genes[gene_name] = parent1.genes[gene_name]
            else:
                child_genes[gene_name] = parent2.genes[gene_name]
        
        child = Gatherer(genes=child_genes)
        return child
    
    def mutate(self, gatherer):
        for gene_name in gatherer.genes:
            if random.random() < MUTATION_RATE:
                # Minimal version: just flip a coin and randomize the gene completely
                min_val, max_val = GENE_RANGES[gene_name]
                current_val = gatherer.genes[gene_name]
                change = random.gauss(0, MUTATION_STRENGTH)
                new_val = current_val + change
                new_val = max(min_val, min(max_val, new_val))
                gatherer.genes[gene_name] = new_val
    
    def create_next_generation(self, population):
        # Evaluate fitness
        fitness_scores = self.evaluate_fitness(population)
        
        # Record statistics
        if fitness_scores:
            best_fitness = fitness_scores[0][1]
            avg_fitness = sum(fitness for _, fitness in fitness_scores) / len(fitness_scores)
            self.fitness_history.append({
                'generation': self.generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            })
            
            # Record trait averages
            all_gatherers = [gatherer for gatherer, _ in fitness_scores]
            trait_averages = {
                'generation': self.generation,
                'avg_speed': sum(g.genes['speed'] for g in all_gatherers) / len(all_gatherers),
                'avg_caution': sum(g.genes['caution'] for g in all_gatherers) / len(all_gatherers),
                'avg_search_pattern': sum(g.genes['search_pattern'] for g in all_gatherers) / len(all_gatherers),
                'avg_efficiency': sum(g.genes['efficiency'] for g in all_gatherers) / len(all_gatherers),
                'avg_cooperation': sum(g.genes['cooperation'] for g in all_gatherers) / len(all_gatherers)
            }
            self.trait_history.append(trait_averages)
        
        # Select survivors
        survivors = self.select_survivors(fitness_scores)
        
        # Create new population
        new_population = []
        
        # Add survivors (reset their state)
        for survivor in survivors:
            new_gatherer = Gatherer(genes=survivor.genes)
            new_population.append(new_gatherer)
        
        # Create offspring to fill remaining slots
        offspring_count = INITIAL_POPULATION - len(survivors)
        for _ in range(offspring_count):
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.generation += 1
        return new_population
    
    def get_population_stats(self, population):
        if not population:
            return {
                'alive_count': 0,
                'total_count': 0,
                'avg_fitness': 0,
                'best_fitness': 0,
                'avg_speed': 0,
                'avg_caution': 0,
                'avg_cooperation': 0
            }
        
        alive_gatherers = [g for g in population if g.alive]
        alive_count = len(alive_gatherers)
        total_count = len(population)
        
        if alive_gatherers:
            fitness_scores = [g.calculate_fitness() for g in alive_gatherers]
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            best_fitness = max(fitness_scores)
            avg_speed = sum(g.genes['speed'] for g in alive_gatherers) / len(alive_gatherers)
            avg_caution = sum(g.genes['caution'] for g in alive_gatherers) / len(alive_gatherers)
            avg_cooperation = sum(g.genes['cooperation'] for g in alive_gatherers) / len(alive_gatherers)
        else:
            # Check all gatherers if none alive
            fitness_scores = [g.calculate_fitness() for g in population]
            avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
            best_fitness = max(fitness_scores) if fitness_scores else 0
            avg_speed = sum(g.genes['speed'] for g in population) / len(population)
            avg_caution = sum(g.genes['caution'] for g in population) / len(population)
            avg_cooperation = sum(g.genes['cooperation'] for g in population) / len(population)
        
        return {
            'alive_count': alive_count,
            'total_count': total_count,
            'avg_fitness': avg_fitness,
            'best_fitness': best_fitness,
            'avg_speed': avg_speed,
            'avg_caution': avg_caution,
            'avg_cooperation': avg_cooperation
        }
    
    def reset(self):
        self.generation = 1
        self.fitness_history = []
        self.trait_history = []

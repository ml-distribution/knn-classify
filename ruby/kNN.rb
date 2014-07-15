require './data_set_reader'
require 'linalg'

module Knn
  class Classifier
    include Linalg
    attr_accessor :dataSet, :labels, :k

    def initialize(d, l, k)
      @dataSet = DMatrix.rows(d)
      @labels, @k = l, k
      @dataSet = normalize(@dataSet)
    end

    def classify(example)
      distances = find_distances(example)
      pairs = pair_up(distances)
      votes = gather_votes(votes, pairs)
      freq = votes.inject(Hash.new(0)) { |h,v| h[v] += 1; h }
      votes.sort_by { |v| freq[v] }.last
    end

    def normalize(set)
      mins, maxes = find_mins_and_maxes(set)
      new_set = set.rows.inject([]) do |new_set, row|
        new_row = []
        row.elems.each_with_index do |cell, idx|
          new_row[idx] = (cell.to_i - mins[idx]) / (maxes[idx] - mins[idx])
        end
        new_set << new_row
      end
      DMatrix.rows(new_set)
    end

    private

    def find_distances(ex)
      distances = @dataSet.rows.inject([]) do |distances, row|
        sum = 0
        row.each_with_index do |el, i|
          sum += ((el - ex[i])**2)
        end
        distances << ::Math.sqrt(sum)
      end
    end

    def find_mins_and_maxes(set)
      mins, maxes = [], []
      set.columns.each_with_index do |column, i|
        maxes[i] = column.to_a.max
        mins[i]  = column.to_a.min
      end
      mins.flatten!
      maxes.flatten!
      [mins, maxes]
    end

    def pair_up(distances)
      pairs = []
      distances.each_with_index do |d, i|
        pairs << [d, @labels[i]]
      end
      pairs = pairs.sort_by { |p| p.first }
      return pairs
    end

    def gather_votes(votes, pairs)
      votes = (0...@k).each.inject([]) do |votes, neighbour|
        votes << pairs[0].last
      end
    end
  end
end
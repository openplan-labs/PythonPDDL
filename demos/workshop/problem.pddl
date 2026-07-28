;; 3 parts, seed 1
(define (problem workshop-3-1)
  (:domain workshop)
  (:objects part1 part2 part3 - part)
  (:init )
  (:goal (and
    (finished part1)
    (finished part2)
    (finished part3))))

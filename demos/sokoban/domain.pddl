;; Sokoban: push boxes onto goal squares, never pull. The grid is encoded as a
;; static `move-dir` relation, so the grounder prunes most of the action space
;; before search even starts. Irreversible pushes make dead ends real, which
;; makes it a great showcase for dead-end detection in the relaxation heuristics.
(define (domain sokoban)
  (:requirements :strips :typing)
  (:types loc box dir - object)

  (:predicates
    (robot-at ?l - loc)
    (box-at ?b - box ?l - loc)
    (clear ?l - loc)
    (move-dir ?from - loc ?to - loc ?d - dir))

  (:action move
    :parameters (?from - loc ?to - loc ?d - dir)
    :precondition (and (robot-at ?from) (move-dir ?from ?to ?d) (clear ?to))
    :effect (and (robot-at ?to) (not (robot-at ?from))
                 (clear ?from) (not (clear ?to))))

  (:action push
    :parameters (?rloc - loc ?bloc - loc ?floc - loc ?d - dir ?b - box)
    :precondition (and (robot-at ?rloc) (box-at ?b ?bloc) (clear ?floc)
                       (move-dir ?rloc ?bloc ?d) (move-dir ?bloc ?floc ?d))
    :effect (and (robot-at ?bloc) (not (robot-at ?rloc)) (clear ?rloc)
                 (box-at ?b ?floc) (not (box-at ?b ?bloc))
                 (not (clear ?floc)))))

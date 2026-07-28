;; Rovers -- exercises disjunctive and quantified preconditions.
(define (domain rovers)
  (:requirements :strips :typing :adl)
  (:types rover waypoint objective - object)
  (:predicates
    (at ?r - rover ?w - waypoint)
    (can-traverse ?a ?b - waypoint)
    (visible ?o - objective ?w - waypoint)
    (imaged ?o - objective)
    (sampled ?w - waypoint)
    (analysed ?w - waypoint)
    (reported ?o - objective))

  (:action navigate
    :parameters (?r - rover ?from ?to - waypoint)
    :precondition (and (at ?r ?from) (can-traverse ?from ?to))
    :effect (and (not (at ?r ?from)) (at ?r ?to)))

  (:action sample
    :parameters (?r - rover ?w - waypoint)
    :precondition (and (at ?r ?w) (not (sampled ?w)))
    :effect (sampled ?w))

  (:action analyse
    :parameters (?w - waypoint)
    :precondition (sampled ?w)
    :effect (analysed ?w))

  ;; An objective can be imaged from any waypoint that sees it.
  (:action image
    :parameters (?r - rover ?o - objective)
    :precondition (exists (?w - waypoint) (and (at ?r ?w) (visible ?o ?w)))
    :effect (imaged ?o))

  ;; Reporting accepts either a picture or a full sample analysis.
  (:action report
    :parameters (?o - objective)
    :precondition (or (imaged ?o)
                      (exists (?w - waypoint)
                              (and (visible ?o ?w) (analysed ?w))))
    :effect (reported ?o)))

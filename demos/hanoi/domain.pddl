;; Towers of Hanoi as STRIPS. One operator, one static "smaller" relation,
;; and a plan length that doubles with every disc -- the cleanest illustration
;; of why heuristic guidance matters.
(define (domain hanoi)
  (:requirements :strips)

  (:predicates
    (clear ?x)
    (on ?x ?y)
    (smaller ?x ?y))

  (:action move
    :parameters (?disc ?from ?to)
    :precondition (and (smaller ?to ?disc)
                       (on ?disc ?from)
                       (clear ?disc)
                       (clear ?to))
    :effect (and (clear ?from)
                 (on ?disc ?to)
                 (not (on ?disc ?from))
                 (not (clear ?to)))))

;; Network routing -- a showcase for :derived-predicates.
;;
;; `connected` is not a fact anyone asserts and no action ever adds it: it is
;; *derived* from the links that happen to be up, by a recursive axiom. Every
;; time an action brings a link up or takes it down, the planner recomputes the
;; transitive closure before looking at the state again.
;;
;; Writing this without derived predicates would mean either enumerating every
;; path as an action or pushing the reachability computation into the goal --
;; which is exactly the situation axioms exist for.
(define (domain network)
  (:requirements :strips :typing :derived-predicates :negative-preconditions)
  (:types node - object)

  (:predicates
    (link ?a - node ?b - node)      ; a cable exists between a and b
    (up ?a - node ?b - node)        ; ...and it is currently switched on
    (connected ?a - node ?b - node) ; derived: a can reach b somehow
    (routed ?a - node ?b - node))   ; we committed to a route

  ;; Reachability: a direct live link, or one hop plus the rest of the path.
  (:derived (connected ?a - node ?b - node)
    (or (up ?a ?b)
        (exists (?m - node) (and (up ?a ?m) (connected ?m ?b)))))

  (:action enable
    :parameters (?a - node ?b - node)
    :precondition (and (link ?a ?b) (not (up ?a ?b)))
    :effect (and (up ?a ?b) (up ?b ?a)))

  (:action disable
    :parameters (?a - node ?b - node)
    :precondition (up ?a ?b)
    :effect (and (not (up ?a ?b)) (not (up ?b ?a))))

  ;; Only legal once the two ends can actually reach each other.
  (:action route
    :parameters (?a - node ?b - node)
    :precondition (connected ?a ?b)
    :effect (routed ?a ?b)))

;; Six nodes wired in a line plus one shortcut, every link initially down.
;;
;;     n1 --- n2 --- n3 --- n4 --- n5 --- n6
;;                    \___________/          (the n3-n5 shortcut)
;;
;; The goal needs n1 to reach n6 and n2 to reach n4. Nothing is connected to
;; start with, so the planner has to switch on a set of links whose transitive
;; closure covers both -- and the cheapest way there uses the shortcut.
(define (problem network-6)
  (:domain network)
  (:objects n1 n2 n3 n4 n5 n6 - node)

  (:init
    (link n1 n2) (link n2 n1)
    (link n2 n3) (link n3 n2)
    (link n3 n4) (link n4 n3)
    (link n4 n5) (link n5 n4)
    (link n5 n6) (link n6 n5)
    (link n3 n5) (link n5 n3))

  (:goal (and
    (routed n1 n6)
    (routed n2 n4))))

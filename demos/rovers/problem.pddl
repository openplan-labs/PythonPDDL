;; 4 objectives, 6 waypoints, seed 11
(define (problem rovers-4-11)
  (:domain rovers)
  (:objects
    rover1 - rover
    wp1 wp2 wp3 wp4 wp5 wp6 - waypoint
    obj1 obj2 obj3 obj4 - objective)
  (:init
    (at rover1 wp1)
    (can-traverse wp1 wp2)
    (can-traverse wp2 wp1)
    (can-traverse wp2 wp3)
    (can-traverse wp2 wp6)
    (can-traverse wp3 wp2)
    (can-traverse wp3 wp4)
    (can-traverse wp4 wp3)
    (can-traverse wp4 wp5)
    (can-traverse wp4 wp6)
    (can-traverse wp5 wp4)
    (can-traverse wp5 wp6)
    (can-traverse wp6 wp5)
    (visible obj1 wp1)
    (visible obj2 wp2)
    (visible obj2 wp3)
    (visible obj3 wp5)
    (visible obj4 wp5))
  (:goal (and
    (reported obj1)
    (reported obj2)
    (reported obj3)
    (reported obj4))))

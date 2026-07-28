;; 3 packages, 5 locations, seed 5
(define (problem numeric-transport-3-5)
  (:domain numeric-transport)
  (:objects
    truck1 - truck
    loc1 loc2 loc3 loc4 loc5 - location
    pkg1 pkg2 pkg3 - package)
  (:init
    (at truck1 loc1)
    (= (fuel truck1) 50)
    (road loc1 loc2)
    (road loc2 loc1)
    (= (distance loc1 loc2) 20)
    (= (distance loc2 loc1) 20)
    (road loc2 loc3)
    (road loc3 loc2)
    (= (distance loc2 loc3) 15)
    (= (distance loc3 loc2) 15)
    (road loc3 loc4)
    (road loc4 loc3)
    (= (distance loc3 loc4) 20)
    (= (distance loc4 loc3) 20)
    (road loc4 loc5)
    (road loc5 loc4)
    (= (distance loc4 loc5) 15)
    (= (distance loc5 loc4) 15)
    (package-at pkg1 loc1)
    (package-at pkg2 loc1)
    (package-at pkg3 loc1))
  (:goal (and
    (package-at pkg1 loc5)
    (package-at pkg2 loc5)
    (package-at pkg3 loc5))))

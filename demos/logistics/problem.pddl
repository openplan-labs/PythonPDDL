;; Two cities, one airplane, one truck per city, three packages to deliver.
;; Every package needs truck -> plane -> truck, so plans are long and the
;; cost metric (flying costs 6, driving 2) genuinely matters.
(define (problem logistics-2c-3p)
  (:domain logistics)
  (:objects
    city1 city2 - city
    truck1 truck2 - truck
    plane1 - airplane
    pkg1 pkg2 pkg3 - package
    apt1 apt2 - airport
    depot1 depot2 - location)

  (:init
    (= (total-cost) 0)
    (in-city apt1 city1) (in-city depot1 city1)
    (in-city apt2 city2) (in-city depot2 city2)
    (at truck1 depot1)
    (at truck2 apt2)
    (at plane1 apt1)
    (at pkg1 depot1)
    (at pkg2 depot1)
    (at pkg3 apt1))

  (:goal (and
    (at pkg1 depot2)
    (at pkg2 apt2)
    (at pkg3 depot2)))

  (:metric minimize (total-cost)))

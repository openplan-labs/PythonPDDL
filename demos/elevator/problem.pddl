;; Six floors, four passengers, all of them going somewhere different.
;; Uninformed search wanders the building; h_FF plans the route.
(define (problem elevator-6f-4p)
  (:domain elevator)
  (:objects
    f0 f1 f2 f3 f4 f5 - floor
    p0 p1 p2 p3 - passenger)

  (:init
    (lift-at f0)
    (origin p0 f1) (destin p0 f4)
    (origin p1 f3) (destin p1 f0)
    (origin p2 f2) (destin p2 f5)
    (origin p3 f5) (destin p3 f1))

  (:goal (and
    (served p0)
    (served p1)
    (served p2)
    (served p3))))

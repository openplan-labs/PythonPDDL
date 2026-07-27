;; A 4x4 room, two boxes, two targets.
;;
;;      p11 p21 p31 p41       @ = robot   $ = box   . = target
;;      p12 p22 p32 p42
;;      p13 p23 p33 p43       @ . . .
;;      p14 p24 p34 p44       . $ . .
;;                            . . $ .
;;                            . * . .      (targets: p31 and p24)
(define (problem sokoban-2box)
  (:domain sokoban)
  (:objects
    p11 p21 p31 p41
    p12 p22 p32 p42
    p13 p23 p33 p43
    p14 p24 p34 p44 - loc
    box1 box2 - box
    right left up down - dir)

  (:init
    ;; --- horizontal adjacency ---
    (move-dir p11 p21 right) (move-dir p21 p31 right) (move-dir p31 p41 right)
    (move-dir p12 p22 right) (move-dir p22 p32 right) (move-dir p32 p42 right)
    (move-dir p13 p23 right) (move-dir p23 p33 right) (move-dir p33 p43 right)
    (move-dir p14 p24 right) (move-dir p24 p34 right) (move-dir p34 p44 right)

    (move-dir p21 p11 left) (move-dir p31 p21 left) (move-dir p41 p31 left)
    (move-dir p22 p12 left) (move-dir p32 p22 left) (move-dir p42 p32 left)
    (move-dir p23 p13 left) (move-dir p33 p23 left) (move-dir p43 p33 left)
    (move-dir p24 p14 left) (move-dir p34 p24 left) (move-dir p44 p34 left)

    ;; --- vertical adjacency ---
    (move-dir p11 p12 down) (move-dir p12 p13 down) (move-dir p13 p14 down)
    (move-dir p21 p22 down) (move-dir p22 p23 down) (move-dir p23 p24 down)
    (move-dir p31 p32 down) (move-dir p32 p33 down) (move-dir p33 p34 down)
    (move-dir p41 p42 down) (move-dir p42 p43 down) (move-dir p43 p44 down)

    (move-dir p12 p11 up) (move-dir p13 p12 up) (move-dir p14 p13 up)
    (move-dir p22 p21 up) (move-dir p23 p22 up) (move-dir p24 p23 up)
    (move-dir p32 p31 up) (move-dir p33 p32 up) (move-dir p34 p33 up)
    (move-dir p42 p41 up) (move-dir p43 p42 up) (move-dir p44 p43 up)

    ;; --- initial configuration ---
    (robot-at p11)
    (box-at box1 p22)
    (box-at box2 p33)

    (clear p21) (clear p31) (clear p41)
    (clear p12) (clear p32) (clear p42)
    (clear p13) (clear p23) (clear p43)
    (clear p14) (clear p24) (clear p34) (clear p44))

  (:goal (and
    (box-at box1 p24)
    (box-at box2 p31))))

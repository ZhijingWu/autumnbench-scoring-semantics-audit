(program
  (= GRID_SIZE 16)

  (object Magnet (: color String) (list (Cell 0 0 color) (Cell 0 1 color)))

  (: fixedMagnet Magnet)
  (= fixedMagnet (initnext (Magnet "red" (Position 7 7)) (prev fixedMagnet)))

  (: mobileMagnet Magnet)
  (= mobileMagnet (initnext (Magnet "blue" (Position 4 7)) (prev mobileMagnet)))

  (: cnt Number)
  (= cnt (initnext 0 (prev cnt)))

  (on left (let (= cnt (+ (prev cnt) 1)) (= mobileMagnet (moveNoCollision (prev "mobileMagnet") -1 0))))
  (on right (let (= cnt (+ (prev cnt) 1)) (= mobileMagnet (moveNoCollision (prev "mobileMagnet") 1 0))))
  (on up (let (= cnt (+ (prev cnt) 1)) (= mobileMagnet (moveNoCollision (prev "mobileMagnet") 0 -1))))
  (on down (let (= cnt (+ (prev cnt) 1)) (= mobileMagnet (moveNoCollision (prev "mobileMagnet") 0 1))))
  (on (adjacentElem (posPole mobileMagnet) (posPole fixedMagnet) 1) (= mobileMagnet (prev "mobileMagnet")))
  (on (adjacentElem (negPole mobileMagnet) (negPole fixedMagnet) 1) (= mobileMagnet (prev "mobileMagnet")))
  (on (in (deltaElem (posPole mobileMagnet) (negPole fixedMagnet)) attractVectors) (
    let
    (= mPos (posPole mobileMagnet))
    (= fNeg (negPole fixedMagnet))
    (= dvec (if (< cnt 5) then (deltaElem mPos fNeg) else (deltaElem fNeg mPos)))
    (if (< cnt 5) then
    (= mobileMagnet (move mobileMagnet (unitVectorSinglePos dvec)))
    else (= fixedMagnet (move fixedMagnet (unitVectorSinglePos dvec))))
  ))
  (on (in (deltaElem (negPole mobileMagnet) (posPole fixedMagnet)) attractVectors) (
    let
    (= dvec (if (< cnt 5) then (deltaElem (negPole mobileMagnet) (posPole fixedMagnet)) else (deltaElem (posPole fixedMagnet) (negPole mobileMagnet))))
    (if (< cnt 5) then
    (= mobileMagnet (move mobileMagnet (unitVectorSinglePos dvec)))
    else (= fixedMagnet (move fixedMagnet (unitVectorSinglePos dvec))))
  ))

  (= posPole (--> (magnet) (head (renderValue magnet))))
  (= negPole (--> (magnet) (tail (renderValue magnet))))

  (= attractVectors (list (Position 0 2) (Position 2 0) (Position -2 0) (Position 0 -2)))
)
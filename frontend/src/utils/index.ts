// you need to install tensorflowjs
// pip install tensorflowjs
//this is use to convert keras model.h5 to tensorflowjs model
// tensorflowjs_converter --input_format=keras model.h5 .

import * as Holistic from "@mediapipe/holistic";
import * as drawing from "@mediapipe/drawing_utils";
export interface Pose {
  faceLandmarks: PoseLandmark[];
  poseLandmarks: PoseLandmark[];
  rightHandLandmarks: PoseLandmark[];
  leftHandLandmarks: PoseLandmark[];
  image: HTMLCanvasElement;
}

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export const FRAMES_FOR_PREDICTION = 20;

export const labelMap = {
  "0": "dance",
  "1": "loud",
  "2": "piano",
  "3": "schedule",
  "4": "tea",
  "5": "angel",
  "6": "all",
  "7": "hit",
  "8": "people",
  "9": "separate",
  "10": "bowl",
  "11": "stand",
  "12": "pull",
  "13": "screwdriver",
  "14": "banana",
  "15": "angry",
  "16": "army",
  "17": "home",
  "18": "depend",
  "19": "center",
  "20": "teacher",
  "21": "actor",
  "22": "fire",
  "23": "sick",
  "24": "steal",
  "25": "divorce",
  "26": "cry",
  "27": "visit",
  "28": "busy",
  "29": "hurt",
  "30": "major",
  "31": "cup",
  "32": "chat",
  "33": "backpack",
  "34": "traffic",
  "35": "closet",
  "36": "store",
  "37": "afternoon",
  "38": "price",
  "39": "believe",
  "40": "again",
  "41": "bicycle",
  "42": "graduate",
  "43": "world",
  "44": "headache",
  "45": "microwave",
  "46": "sandwich",
  "47": "before",
  "48": "lunch",
  "49": "small",
  "50": "love",
  "51": "lost",
  "52": "stay",
  "53": "book",
  "54": "day",
  "55": "humble",
  "56": "lazy",
  "57": "easy",
  "58": "near",
  "59": "corn",
  "60": "newspaper",
  "61": "can",
  "62": "last",
  "63": "university",
  "64": "fail",
  "65": "kiss",
  "66": "afraid",
  "67": "movie",
  "68": "month",
  "69": "table",
  "70": "rich",
  "71": "college",
  "72": "australia",
  "73": "right",
  "74": "give",
  "75": "take up",
  "76": "chicken",
  "77": "hour",
  "78": "sad",
  "79": "decrease",
  "80": "class",
  "81": "mushroom",
  "82": "destroy",
  "83": "lobster",
  "84": "sit",
  "85": "save",
  "86": "eat",
  "87": "dolphin",
  "88": "comfortable",
  "89": "football",
  "90": "mix",
  "91": "today",
  "92": "cow",
  "93": "street",
  "94": "socks",
  "95": "wife",
  "96": "sister",
  "97": "clock",
  "98": "test",
  "99": "since",
  "100": "develop",
  "101": "slow",
  "102": "can't",
  "103": "negative",
  "104": "school",
  "105": "magazine",
  "106": "soda",
  "107": "biology",
  "108": "shock",
  "109": "minute",
  "110": "deer",
  "111": "brother",
  "112": "fishing",
  "113": "rabbit",
  "114": "together",
  "115": "tomato",
  "116": "alarm",
  "117": "fence",
  "118": "how_many",
  "119": "turtle",
  "120": "email",
  "121": "check",
  "122": "bacon",
  "123": "want",
  "124": "in",
  "125": "house",
  "126": "coffee",
  "127": "where",
  "128": "run",
  "129": "easter",
  "130": "midnight",
  "131": "drop",
  "132": "inform",
  "133": "quiet",
  "134": "ugly",
  "135": "drive",
  "136": "goal",
  "137": "laugh",
  "138": "decide",
  "139": "less",
  "140": "jump",
  "141": "arrive",
  "142": "hard",
  "143": "glasses",
  "144": "excited",
  "145": "bed",
  "146": "belt",
  "147": "change",
  "148": "sell",
  "149": "toast",
  "150": "feel",
  "151": "homework",
  "152": "wood",
  "153": "autumn",
  "154": "cracker",
  "155": "towel",
  "156": "couch",
  "157": "have",
  "158": "first",
  "159": "with",
  "160": "born",
  "161": "from",
  "162": "basement",
  "163": "start",
  "164": "bird",
  "165": "appointment",
  "166": "tough",
  "167": "ocean",
  "168": "star",
  "169": "happy",
  "170": "egg",
  "171": "equal",
  "172": "skirt",
  "173": "church",
  "174": "famous",
  "175": "follow",
  "176": "roommate",
  "177": "birthday",
  "178": "full",
  "179": "therapy",
  "180": "rain",
  "181": "medicine",
  "182": "nothing",
  "183": "blue",
  "184": "cool",
  "185": "good",
  "186": "car",
  "187": "river",
  "188": "team",
  "189": "office",
  "190": "chair",
  "191": "buy",
  "192": "salad",
  "193": "mean",
  "194": "awful",
  "195": "measure",
  "196": "happen",
  "197": "game",
  "198": "shoes",
  "199": "deaf",
  "200": "country",
  "201": "hello",
  "202": "friend",
  "203": "dessert",
  "204": "help",
  "205": "train",
  "206": "animal",
  "207": "principal",
  "208": "pants",
  "209": "match",
  "210": "knife",
  "211": "umbrella",
  "212": "name",
  "213": "soup",
  "214": "ball",
  "215": "but",
  "216": "which",
  "217": "germany",
  "218": "dollar",
  "219": "butter",
  "220": "push",
  "221": "science",
  "222": "party",
  "223": "camping",
  "224": "underwear",
  "225": "freeze",
  "226": "baby",
  "227": "wedding",
  "228": "basketball",
  "229": "swim",
  "230": "paint",
  "231": "soccer",
  "232": "answer",
  "233": "motorcycle",
  "234": "spanish",
  "235": "pillow",
  "236": "pet",
  "237": "wet",
  "238": "different",
  "239": "classroom",
  "240": "spoon",
  "241": "vacation",
  "242": "secretary",
  "243": "garage",
  "244": "explain",
  "245": "confused",
  "246": "surprise",
  "247": "flag",
  "248": "marry",
  "249": "worry",
  "250": "poor",
  "251": "cake",
  "252": "building",
  "253": "tree",
  "254": "dictionary",
  "255": "total",
  "256": "copy",
  "257": "argue",
  "258": "retire",
  "259": "dinner",
  "260": "go",
  "261": "bath",
  "262": "play",
  "263": "young",
  "264": "jacket",
  "265": "around",
  "266": "win",
  "267": "light",
  "268": "snow",
  "269": "brave",
  "270": "discuss",
  "271": "sew",
  "272": "some",
  "273": "potato",
  "274": "meet",
  "275": "window",
  "276": "bad",
  "277": "picture",
  "278": "law",
  "279": "chemistry",
  "280": "all day",
  "281": "far",
  "282": "close",
  "283": "girlfriend",
  "284": "key",
  "285": "die",
  "286": "doctor",
  "287": "rest",
  "288": "make",
  "289": "sunday",
  "290": "salt",
  "291": "english",
  "292": "kitchen",
  "293": "sweetheart",
  "294": "careful",
  "295": "earthquake",
  "296": "plate",
  "297": "mustache",
  "298": "floor",
  "299": "nurse",
  "300": "research",
  "301": "shop",
  "302": "week",
  "303": "blanket",
  "304": "many",
  "305": "clothes",
  "306": "bread",
  "307": "hoddog",
  "308": "business",
  "309": "bear",
  "310": "pencil",
  "311": "cherry",
  "312": "winter",
  "313": "sign",
  "314": "meeting",
  "315": "intersection",
  "316": "date",
  "317": "elevator",
  "318": "money",
  "319": "door",
  "320": "accident",
  "321": "hockey",
  "322": "spring",
  "323": "boat",
  "324": "hamburger",
  "325": "boyfriend",
  "326": "letter",
  "327": "city",
  "328": "fat",
  "329": "night",
  "330": "remember",
  "331": "study",
  "332": "finish",
  "333": "mountain",
  "334": "get up",
  "335": "candle",
  "336": "son",
  "337": "dark",
  "338": "plus",
  "339": "lie",
  "340": "daughter",
  "341": "computer",
  "342": "earn",
  "343": "guitar",
  "344": "use",
  "345": "awkward",
  "346": "now",
  "347": "butterfly",
  "348": "get",
  "349": "become",
  "350": "more",
  "351": "walk",
  "352": "artist",
  "353": "husband",
  "354": "squirrel",
  "355": "example",
  "356": "glass",
  "357": "free",
  "358": "write",
  "359": "student",
  "360": "blame",
  "361": "watermelon",
  "362": "enter",
  "363": "sweater",
  "364": "cook",
  "365": "new",
  "366": "involve",
  "367": "long",
  "368": "pass",
  "369": "language",
  "370": "cute",
  "371": "tall",
  "372": "friendly",
  "373": "number",
  "374": "family",
  "375": "vomit",
  "376": "education",
  "377": "monkey",
  "378": "year",
  "379": "cheese",
  "380": "important",
  "381": "big",
  "382": "not want",
  "383": "thank you",
  "384": "cheap",
  "385": "cost",
  "386": "tired",
  "387": "what",
  "388": "here",
  "389": "island",
  "390": "math",
  "391": "draw",
  "392": "early",
  "393": "after",
  "394": "interesting",
  "395": "most",
  "396": "room",
  "397": "noon",
  "398": "short",
  "399": "librarian",
  "400": "internet",
  "401": "weekend",
  "402": "introduce",
  "403": "teach",
  "404": "leave",
  "405": "gone",
  "406": "allow",
  "407": "exercise",
  "408": "gray",
  "409": "time",
  "410": "maybe",
  "411": "baseball",
  "412": "dress",
  "413": "clean",
  "414": "grandmother",
  "415": "fun",
  "416": "tiger",
  "417": "stop",
  "418": "strong",
  "419": "break",
  "420": "sometimes",
  "421": "nervous",
  "422": "counselor",
  "423": "pie",
  "424": "both",
  "425": "interest",
  "426": "join",
  "427": "camp",
  "428": "cold",
  "429": "read",
  "430": "challenge",
  "431": "cookie",
  "432": "vote",
  "433": "list",
  "434": "paper",
  "435": "weak",
  "436": "forget",
  "437": "weather",
  "438": "drink",
  "439": "grapes",
  "440": "watch",
  "441": "fish",
  "442": "often",
  "443": "popcorn",
  "444": "machine",
  "445": "wait",
  "446": "lawyer",
  "447": "boots",
  "448": "crash",
  "449": "fine",
  "450": "show",
  "451": "hate",
  "452": "hurry",
  "453": "next",
  "454": "gossip",
  "455": "pay",
  "456": "shirt",
  "457": "nice",
  "458": "like",
  "459": "tie",
  "460": "children",
  "461": "build",
  "462": "continue",
  "463": "practice",
  "464": "music",
  "465": "about",
  "466": "excuse",
  "467": "plan",
  "468": "when",
  "469": "beard",
  "470": "bedroom",
  "471": "same",
  "472": "bake",
  "473": "disappear",
  "474": "ready",
  "475": "beautiful",
  "476": "halloween",
  "477": "learn",
  "478": "japan",
  "479": "fast",
  "480": "spider",
  "481": "breakfast",
  "482": "cheat",
  "483": "chocolate",
  "484": "fight",
  "485": "interpreter",
  "486": "work",
  "487": "america",
  "488": "expensive",
  "489": "sweep",
  "490": "soap",
  "491": "meat",
  "492": "rude",
  "493": "arrogant",
  "494": "enjoy",
  "495": "how",
  "496": "live",
  "497": "morning",
  "498": "girl",
  "499": "ticket",
};

export const connect = (
  ctx: CanvasRenderingContext2D,
  connectors: Array<[Holistic.NormalizedLandmark, Holistic.NormalizedLandmark]>
): void => {
  const canvas = ctx.canvas;
  for (const connector of connectors) {
    const from = connector[0];
    const to = connector[1];
    if (from && to) {
      if (
        from.visibility &&
        to.visibility &&
        (from.visibility < 0.1 || to.visibility < 0.1)
      ) {
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
      ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
      ctx.stroke();
    }
  }
};
const IGNORED_BODY_LANDMARKS = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22,
];

function drawBody(
  landmarks: PoseLandmark[],
  ctx: CanvasRenderingContext2D
): void {
  const filteredLandmarks = Array.from(landmarks);
  for (const l of IGNORED_BODY_LANDMARKS) {
    delete filteredLandmarks[l];
  }

  drawing.drawConnectors(ctx, filteredLandmarks, Holistic.POSE_CONNECTIONS, {
    color: "#CC0000",
  });
  drawing.drawLandmarks(ctx, filteredLandmarks, {
    color: "#CC0000",
    fillColor: "#FF0000",
  });
}
function drawNeck(
  landmarks: PoseLandmark[],
  ctx: CanvasRenderingContext2D,
  lineColor: string
) {
  // get the left and right shoulder landmarks
  const leftShoulder = landmarks[Holistic.POSE_LANDMARKS.LEFT_SHOULDER];
  const rightShoulder = landmarks[Holistic.POSE_LANDMARKS.RIGHT_SHOULDER];
  const nose = landmarks[Holistic.POSE_LANDMARKS.NOSE];
  //lower the nose by a percentage to react to the chin
  nose.y = nose.y + 0.1;

  // draw a vertical line between the two shoulders to the head
  ctx.beginPath();
  const x = (leftShoulder.x + rightShoulder.x) / 2;
  const y = (leftShoulder.y + rightShoulder.y) / 2;
  ctx.moveTo(x * ctx.canvas.width, y * ctx.canvas.height);
  ctx.lineTo(nose.x * ctx.canvas.width, nose.y * ctx.canvas.height);
  ctx.strokeStyle = lineColor;
  ctx.stroke();
}

function drawHand(
  landmarks: PoseLandmark[],
  ctx: CanvasRenderingContext2D,
  lineColor: string,
  dotColor: string,
  dotFillColor: string
): void {
  const fingerColors = ["#FF0000", "#FFA500", "#FFFF00", "#008000", "#0000FF"];
  drawConnect([[landmarks[0], landmarks[9]]], ctx, fingerColors[2]); //middle finger
  drawConnect([[landmarks[0], landmarks[13]]], ctx, fingerColors[3]); //ring finger

  drawing.drawConnectors(ctx, landmarks, Holistic.HAND_CONNECTIONS, {
    color: (info) => {
      const ignore = [8, 12, 16];
      if (ignore.includes(info.index ?? 0)) return "transparent";
      return fingerColors[Math.floor((info.index ?? 0) / 4)];
    },
  });

  //   for (let i = 0; i < 5; i++) {
  //     const fingerLandmarks = landmarks.slice(i * 4 + 1, i * 4 + 5); // get landmarks for current finger
  //     drawing.drawLandmarks(ctx, fingerLandmarks, {
  //       color: fingerColors[i], // set color to color of current finger
  //       fillColor: dotFillColor,
  //       lineWidth: 2,
  //       radius: (landmark) => {
  //         return drawing.lerp(1, -0.15, 0.01, 10, 1);
  //       },
  //     });
  //   }
}

function drawFace(
  landmarks: PoseLandmark[],
  ctx: CanvasRenderingContext2D
): void {
  //   drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_TESSELATION, {
  //     color: "#C0C0C070",
  //     lineWidth: 1,
  //   });
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_RIGHT_EYE, {
    color: "#FF3030",
  });
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_RIGHT_EYEBROW, {
    color: "#FF3030",
  });
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_LEFT_EYE, {
    color: "#30FF30",
  });
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_LEFT_EYEBROW, {
    color: "#30FF30",
  });
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_FACE_OVAL, {
    color: "#E0E0E0",
  });
  drawing.drawConnectors(ctx, landmarks, Holistic.FACEMESH_LIPS, {
    color: "#E0E0E0",
  });
}

function drawConnect(
  connectors: PoseLandmark[][],
  ctx: CanvasRenderingContext2D,
  color = "#CC0000"
): void {
  for (const connector of connectors) {
    const from = connector[0];
    const to = connector[1];
    if (from && to) {
      if (
        from.visibility &&
        to.visibility &&
        (from.visibility < 0.1 || to.visibility < 0.1)
      ) {
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(from.x * ctx.canvas.width, from.y * ctx.canvas.height);
      ctx.lineTo(to.x * ctx.canvas.width, to.y * ctx.canvas.height);
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  }
}

function drawElbowHandsConnection(
  pose: Pose,
  ctx: CanvasRenderingContext2D
): void {
  ctx.lineWidth = 5;

  if (pose.rightHandLandmarks) {
    ctx.strokeStyle = "#CC0000";
    drawConnect(
      [
        [
          pose.poseLandmarks[Holistic.POSE_LANDMARKS.RIGHT_ELBOW],
          pose.rightHandLandmarks[0],
        ],
      ],
      ctx
    );
  }

  if (pose.leftHandLandmarks) {
    ctx.strokeStyle = "#FF0000";
    drawConnect(
      [
        [
          pose.poseLandmarks[Holistic.POSE_LANDMARKS.LEFT_ELBOW],
          pose.leftHandLandmarks[0],
        ],
      ],
      ctx
    );
  }
}

export function draw(
  ctx: CanvasRenderingContext2D,
  pose: Holistic.Results,
  drawImage = true
) {
  if (!ctx) return;
  ctx.save();
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (drawImage && pose.image) {
    ctx.drawImage(pose.image, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  if (pose.poseLandmarks) {
    drawBody(pose.poseLandmarks, ctx);
    drawElbowHandsConnection(pose as Pose, ctx);
    drawNeck(pose.poseLandmarks, ctx, "#CC0000");
  }

  if (pose.leftHandLandmarks) {
    drawHand(pose.leftHandLandmarks, ctx, "#CC0000", "#FF0000", "#CC0000");
  }

  if (pose.rightHandLandmarks) {
    drawHand(pose.rightHandLandmarks, ctx, "#CC0000", "#CC0000", "#FF0000");
  }

  if (pose.faceLandmarks) {
    drawFace(pose.faceLandmarks, ctx);
  }

  ctx.restore();
}

export const getActionFrames = async (text: string) => {
  //load json file from public folder in react app
  const json = await fetch("/keypoints_reshaped.json");
  const data = await json.json();
  const frames = data[text];
  return frames;
};

export async function drawFramesAndCreateVideo(
  sentence: string[],
  speed: number,
  fps = 30
) {
  return new Promise<string>(async (resolve, reject) => {
    const canvas = document.createElement("canvas");
    //canvas aspect ratio is 16:9
    canvas.width = 1280;
    canvas.height = 720;
    //canvas background color
    canvas.style.backgroundColor = "red";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Get the frames from getActionFrames function
    const promises = sentence.map((text) => getActionFrames(text));
    const frames = await Promise.all(promises);
    //merge frames
    const mergedFrames = frames.reduce((acc, cur) => {
      return acc.concat(cur);
    }, []);

    // Create a new MediaRecorder instance
    const stream = canvas.captureStream();
    const chunks: Blob[] = [];
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      chunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      const blob = new Blob(chunks, { type: "video/mp4" });
      const videoURL = URL.createObjectURL(blob);
      resolve(videoURL);
    };

    // Start recording
    mediaRecorder.start();

    // Draw the frames
    for (let i = 0; i < mergedFrames.length; i++) {
      draw(ctx, mergedFrames[i], false);
      await new Promise((resolve) => setTimeout(resolve, 1000 / fps / speed));
    }

    // Stop recording
    mediaRecorder.stop();
  });
}

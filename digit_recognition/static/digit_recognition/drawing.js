let wrapper = document.getElementById('wrapper');
let canvas = document.getElementById('drawing_board');
let eraser = document.getElementById('eraser');
let pencil = document.getElementById('pencil');
let select = document.getElementById('select');
let trash = document.getElementById('trash');
let undo = document.getElementById('undo');
let redo = document.getElementById('redo');
let collapser = document.getElementById('collapser');

let drawer = canvas.getContext('2d');
let body = document.body;

let active = false;
let drawing = false;
let tool = 'select';

let prev_x = 0;
let prev_y = 0;
let length = 0;

const square_clear = 30;
const size = 477;

const history_size = 10;
let action_ptr = 0;
let action_stack = Array(history_size).fill(null);
action_stack[0] = drawer.getImageData(0, 0, size, size);


colors = ['red', 'green', 'blue', 'yellow'];

drawer.fillStyle = 'red';
canvas.style.border = '0px solid darkgray';

canvas.addEventListener('click', toggle, {capture: true});
canvas.addEventListener('mousedown', () => down(false));
canvas.addEventListener('mouseup', up);
canvas.addEventListener('mousemove', () => draw(false));
canvas.addEventListener('mouseout', mouseout);
canvas.addEventListener('mouseover', mouseover);

canvas.addEventListener('touchstart', () => down(true));
canvas.addEventListener('touchend', up);
canvas.addEventListener('touchmove', () => draw(true));

body.addEventListener('click', deactivate);

select.addEventListener('click', () => change_tool(select), {capture: true});
pencil.addEventListener('click', () => change_tool(pencil), {capture: true});
eraser.addEventListener('click', () => change_tool(eraser), {capture: true});
trash.addEventListener('click', trashcan, {capture: true});
undo.addEventListener('click', f_undo, {capture: true});
redo.addEventListener('click', f_redo, {capture: true});

function f_undo() {
    event.stopPropagation();
    if (action_ptr-1 >= 0 && action_stack[(action_ptr-1)] !== null) {
        action_ptr--;
        drawer.putImageData(action_stack[action_ptr], 0, 0);
    }
}

function f_redo() {
    event.stopPropagation();
    if (action_stack[(action_ptr+1)] !== null && action_ptr+1 <= action_stack.length) {
        action_ptr++;
        drawer.putImageData(action_stack[action_ptr], 0, 0);
    }
}


function trashcan() {
    event.stopPropagation();
    drawer.clearRect(0, 0, size, size);
    toggle(canvas);
    save_data();
}


function toggle() {
    event.stopPropagation();
    if (active === false) {
        canvas.classList.replace('select', tool);
        collapser.classList.toggle('collapsed');
        $("#"+wrapper.id).animate({borderWidth: '4px'}, 'fast');
        active = !active;
        console.log('activated')
    }
}

function deactivate() {
    if (active === true) {
        canvas.classList.replace(tool, 'select');
        active = !active;
        collapser.classList.toggle('collapsed');
        $("#"+wrapper.id).animate({borderWidth: '0px'}, 'fast');
        console.log('deactivated')
    }
}

function change_tool(object) {
    event.stopPropagation();
    $("#"+object.id).button('toggle');
    if (active === false) {
        toggle(canvas)
    }
    if (tool !== object.id) {
        canvas.classList.replace(tool, object.id);
        tool = object.id;
        console.log(tool)
    }
}

function down(touch_mode) {
    if (active === true && (tool === 'pencil' || tool === 'eraser')) {
        length = 0;
        collapser.classList.toggle('collapsed');
        drawing = true;
        if (touch_mode === true) {
            prev_x = event.changedTouches[0].clientX - canvas.getBoundingClientRect().left;
            prev_y = event.changedTouches[0].clientY - canvas.getBoundingClientRect().top;
            event.preventDefault();
        } else {
            prev_x = event.clientX - canvas.getBoundingClientRect().left;
            prev_y = event.clientY - canvas.getBoundingClientRect().top;
        }
    }
}

function up(){
    if (drawing === true) {
        collapser.classList.toggle('collapsed');
        if (length > 1) {
            save_data();
        }
        length = 0;
        drawing = false;
    }
}

function mouseover() {
     if (active) {
         prev_x = event.clientX - canvas.getBoundingClientRect().left;
         prev_y = event.clientY - canvas.getBoundingClientRect().top;
     }
}

function mouseout() {
    if (tool === 'pencil' && drawing === true) {
        drawer.stroke();
    }
}

function draw(touch_mode) {

    if (active) {
        let x;
        let y;
        if (drawing) {
            length++;
            if (touch_mode === true) {
                x = event.changedTouches[0].clientX - canvas.getBoundingClientRect().left;
                y = event.changedTouches[0].clientY - canvas.getBoundingClientRect().top;
                event.preventDefault();
            } else {
                x = event.clientX - canvas.getBoundingClientRect().left;
                y = event.clientY - canvas.getBoundingClientRect().top;
            }

            if (tool === 'pencil') {
                drawer.strokeStyle = 'black';
                drawer.lineCap = 'round';

                drawer.beginPath();
                drawer.moveTo(prev_x, prev_y);
                drawer.lineTo(x, y);
                drawer.lineWidth = 5;
                drawer.stroke();
            } else {
                let tmp, length;

                let x1 = prev_x, x2 = x, y1 = prev_y, y2 = y;

                // swap coordinate pairs if x-coordinates are RTL to make them LTR
                if (x2 < x1) {
                    tmp = x1;
                    x1 = x2;
                    x2 = tmp;
                    tmp = y1;
                    y1 = y2;
                    y2 = tmp;
                }

                length = dist(x1, y1, x2, y2) + 5;

                drawer.save();
                drawer.translate(x1, y1);
                drawer.rotate(Math.atan2(y2 - y1, x2 - x1));
                drawer.clearRect(0, -square_clear/2, length, square_clear);
                drawer.restore();
            }
            prev_x = x;
            prev_y = y;
            console.log(x + ' ' + y)
        }
    }
}

function dist(x1,y1,x2,y2) {
	x2-=x1;
	y2-=y1;
	return Math.sqrt((x2*x2) + (y2*y2));
}

function save_data() {
    if (action_stack[action_ptr+1] !== null) {
        action_stack.fill(null, action_ptr+1);
    }
    if (action_ptr+1 >= action_stack.length) {
        let temp = action_stack.slice(history_size/2, history_size);
        action_stack.fill(null);
        action_stack = temp.concat(Array(history_size/2).fill(null));
        action_ptr = history_size/2;
        action_stack[action_ptr] = drawer.getImageData(0, 0, size, size);
    } else {
        action_stack[(action_ptr + 1)] = drawer.getImageData(0, 0, size, size);
        action_ptr++;
    }
}


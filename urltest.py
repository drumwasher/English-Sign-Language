from flask import Flask, render_template, request
from flask_ask import Ask, statement
from RPLCD.i2c import CharLCD
from gpiozero import LED
import os
from time import sleep

# 0: on/off 1:blink 2~6:상위~하위 비트
led=[LED(17),LED(27),LED(26),LED(19),LED(13),LED(6),LED(5),LED(11)]

lcd = CharLCD('PCF8574', 0x27)
lcd.backlight_enabled=False

sentence=''
s_flag=0 # on/off 플래그
b_flag=0 # blink 플래그

led[0].on()

app=Flask(__name__)
ask=Ask(app,'/')

@app.route('/')
def homepage():
	return 'Url test homepage'

@app.route('/method',methods=['GET'])
def method():
	global sentence, flag, position,s_flag,b_flag

	name = request.args.get('stage')
	if name == 'clear':
		lcd.clear()
		sentence=''

	elif name == 'space':
		lcd.write_string(' ')
		sentence+=' '

	elif name == 'enter':
		lcd.clear()
		lcd.write_string('Playing')
		os.system('python synthesize_text.py --text "{}"'.format(sentence))
		os.system('mpg321 output.mp3')
		lcd.clear()
		lcd.write_string('Done')
		sleep(0.3)
		lcd.clear()
		sentence=''

	elif name == 'back':
		if lcd.cursor_pos==(1,0):
			lcd.cursor_pos=(0,16)

		pos=lcd.cursor_pos
		pos=(pos[0],pos[1]-1)
		lcd.cursor_pos=pos
		lcd.write_string(' ')
		lcd.cursor_pos=pos
		sentence=sentence[:-1]

	elif name == 'state':
		s_flag+=1
		s_flag%=2
		sentence=''
		if s_flag==1 : 
			lcd.cursor_mode='blink'
			lcd.backlight_enabled=True
			b_flag=1
		else :
			lcd.clear()
			lcd.cursor_mode='hide'
			lcd.backlight_enabled=False
		
	else : #print alphabet
		lcd.write_string(name)
		sentence+=name

	#on/off(gree)
	if s_flag==0 :
		led[0].on()
	else : led[0].off()

	#blink(yellow)
	if s_flag==1 & b_flag==1:
		led[1].blink()
		b_flag=0
	elif s_flag==0 & b_flag==0:
		led[1].off()
	
	#bit count
	if len(sentence) <= 32 :
		str1=[x for x in bin(len(sentence))]
		del str1[0:2]
		for i in range(6-len(str1)):
			str1.insert(0,'0')
		for i in range(len(str1)): led[i+2].value= int(str1[i])
	
	#next line	
	if lcd.cursor_pos==(0,16):
			lcd.crlf()	

	return 'good'



if __name__=="__main__":
	app.run(port=5000, debug=True)




import pandas as pd
import streamlit as st
from main import import_data, train_best_model, make_pred, create_input, compare_position

st.set_page_config(page_title='🏈 Future Football Star 🏈',
                   layout='wide',
                   initial_sidebar_state='expanded')

if __name__ == '__main__':

    #Buid the dataset
    x_train, x_test, y_train, y_test, sc, columns = import_data()

    #Build Model
    model, cr, acc = train_best_model(x_train, y_train, x_test, y_test)

    #Streamlit app
    st.markdown('<p style="color: white; text-align: center; font-size: 40px;">🏈 Future Football Star 🏈</p>',
                unsafe_allow_html=True)
    st.markdown('<p style="color: white; text-align: center; font-size: 20px;">Input Your Football Stats To Get Your Chances of Being Drafted</p>',
                unsafe_allow_html=True)
    st.markdown('# \n')

    # Take user input
    st.sidebar.header('Football Stats')
    with st.sidebar:
        st.divider()
        positions = ['QB', 'RB', 'WR', 'C', 'CB', 'DB',
                     'DE', 'DL', 'DT', 'EDGE', 'FB', 'ILB',
                     'K', 'LB', 'LS', 'OG', 'OL', 'OLB', 'OT', 'P', 'S', 'TE']
        year = st.selectbox('Year of Projected Draft Date', options=[num for num in range(2024,2100)], index=0)
        st.divider()

        position = st.selectbox('What Position Do You Play?', options=positions, index=0)
        st.divider()

        height = st.slider(label='Height (in)', min_value=60, max_value=96, step=1, value=74)
        weight = st.slider(label='Weight (lbs)', min_value=0, max_value=450, value=220)
        st.divider()

        st.write('Measured Events')
        dash = st.slider(label='40yd Dash (s)', min_value=3.0, max_value=10.0, value=4.8, step=0.01)
        with st.expander(label='How to Measure 40yd Dash'):
            st.write("""To complete a 40-yard dash, an athlete starts from a stationary position 
            and runs as fast as possible for 40 yards (36.6 meters), typically timed with a stopwatch
             to measure speed and acceleration over a short distance.""")

        vertical = st.slider(label='Vertical (in)', min_value=10.0, max_value=50., value=32., step=0.5)
        with st.expander(label='How to Measure Vertical Jump'):
            st.write("""To measure vertical jump, stand next to a wall with a piece of chalk.
             Jump as high as possible, marking the wall at the highest point you touch. Measure
              the distance from the ground to the mark for your vertical jump height.""")

        bench = st.selectbox('Bench Press Reps of 225lbs', options=[num for num in range(61)], index=2)
        with st.expander(label='How to Measure Bench Press'):
            st.write("""For bench reps, lie on a flat bench and lift a barbell with weights,
             lowering it to your chest and pushing it back up. Count each successful lift
              as one rep. NFL combines use a standardized 225 lbs and measure reps until exhaustion.""")

        jump = st.slider(label='Broad Jump (in)', min_value=0., max_value=150., value=110., step=1.0)
        with st.expander(label='How to Measure Broad Jump'):
            st.write("""Measure the broad jump by starting from a standstill, jumping forward
             as far as possible, and landing on both feet. Measure the distance from the takeoff
              line to the back of the heels at landing.""")

        cone = st.slider(label='3Cone Time (s)', min_value=0., max_value=15., value=7., step=.01)
        with st.expander(label='How to Measure 3Cone'):
            st.write("""Set up three cones in an L-shape. Start at the base cone, run 5 yards
             to the second cone, change direction and sprint 10 yards back to the first cone,
              then change direction again and sprint around the second cone. Finally, run around
               the third cone, sprinting back to the starting cone.""")

        shuttle = st.slider(label='Shuttle Time (s)', min_value=0., max_value=10., value=4.3, step=.01)
        with st.expander(label='How to Measure Shuttle Run'):
            st.write("""Begin in a three-point stance with one hand down. Sprint 5 yards
             to the left cone, touch it with your hand, sprint 10 yards to the right cone, touch
              it with your hand, and finally sprint back through the starting line.""")

    # Change input to proper format
    input_df = create_input(year, position, height, weight, dash, vertical, bench, jump, cone, shuttle, sc)

    # Predict if they would be drafted or not
    pred, proba = make_pred(input_df, model)

    #Include how to perform each test if needed in expanders

    col_1, col_2 = st.columns([4,1])

    #Let them know the average results for players of the same position in those stats (show side by side bar graph)
    with col_1:
        st.markdown(f'## User Compared to Drafted {position}\'s')
        fig = compare_position(position, height, weight, dash, vertical, bench, jump, cone, shuttle)
        st.plotly_chart(fig)
    #Tell them what they could improve

    #Show model accuracy and classification report
    with col_2:
        st.markdown('<p style="color: white; text-align: center; font-size: 36px;">Draft Prediction</p>',
                    unsafe_allow_html=True)
        # st.write(f'With {round(proba * 100, 2)}% Certainty')
        st.markdown('<p style="text-align: center; font-size: 18px;">'
                    f'With {round(proba * 100, 2)}% Certainty'
                    '</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: white; text-align: center; font-size: 18px;">Are You a Draft Prospect?</p>',
                    unsafe_allow_html=True)
        if pred == 'Drafted':
            st.markdown('<p style="color: green; text-align: center; font-size: 34px;">Drafted</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: red; text-align: center; font-size: 34px;">Undrafted</p>',
                        unsafe_allow_html=True)
        st.markdown('\n')
        st.markdown('<p style="color: white; text-align: center; font-size: 18px;">Model Info</p>',
                    unsafe_allow_html=True)
        with st.expander(label='Model Report'):
            st.write('Random Forest Classifier')
            st.write(f'Accuracy: {round(acc * 100, 2)}%')
            st.write(f'Precision for Positive class: {round(cr["1"]["precision"], 2)}')
            st.write(f'Recall for Positive class: {round(cr["1"]["recall"], 2)}')
            st.write(f'F-1 Score for Positive class: {round(cr["1"]["f1-score"], 2)}')
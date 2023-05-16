def moveStraight(self, agent_host, factor, world_state):
        flag = False
        move_speed = factor * 2
        agent_host.sendCommand('move {}'.format(move_speed))
        isAlive = True
        last_timeAlive = None
        while isAlive and flag is False:
            latest_ws = agent_host.peekWorldState()
            print(f'TurnDegree, Latest world state is: {latest_ws}')
            # If there are some new observations
            if latest_ws.number_of_observations_since_last_state > 0:
                obs_text = latest_ws.observations[-1].text
                obs = json.loads(obs_text)
                print(f'Peek World State is:{obs}')
                current_ZPos = float(obs[u'ZPos'])
                current_XPos = float(obs[u'XPos'])
                current_manhattan = current_XPos + current_ZPos
                # use manhattan distance to calculate distance between current and target
                # manhattan distance: x + z 
                # 1 gaussian distance ~ 1.414 manhattan distance
                target_manhattan = current_ZPos + current_XPos + 1.414
                print(f'Init Current XPos is {current_XPos}, ZPos is {current_ZPos}, target manhattan is {target_manhattan}')
                while isAlive and abs(current_manhattan - target_manhattan) > 0.1:
                    time.sleep(0.1)
                    agent_host.sendCommand('move {}'.format(move_speed))
                    latest_ws = agent_host.peekWorldState()
                    # If there are some new observations
                    if latest_ws.number_of_observations_since_last_state > 0:
                        obs_text = latest_ws.observations[-1].text
                        obs = json.loads(obs_text)
                        # print(f'Peek World State is:{obs}')
                        current_ZPos = float(obs[u'ZPos'])
                        current_XPos = float(obs[u'XPos'])
                        current_manhattan = current_XPos + current_ZPos
                        timeAlive = obs[u'TimeAlive']
                        if timeAlive == last_timeAlive:
                            isAlive = False
                        last_timeAlive = timeAlive
                        agent_host.sendCommand('move {}'.format(move_speed))
                        print(obs)
                        print(f'Init Current XPos is {current_XPos}, ZPos is {current_ZPos}, target manhattan is {target_manhattan}')
                flag = True
        print(f'move straight {factor} success!')
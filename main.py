import sys
from fullAstaralg import RingDestroyerGame


def main():
    game = RingDestroyerGame()
    
    # Обрабатываем начальный ввод
    game.process_initial_input()
    
    # Основной игровой цикл
    try:
        while True:
            # Проверяем, не достигли ли мы цели
            if game.should_stop():
                final_command = game.get_final_command()
                print(final_command)
                sys.stdout.flush()
                break
                
            # Получаем следующую команду с помощью A*
            command = game.execute_astar_move()
            
            if not command:
                # Если путь не найден
                print("e -1")
                sys.stdout.flush()
                break
                
            # Отправляем команду и проверяем завершение
            should_exit = game.send_command(command)
            if should_exit:
                break
                
    except Exception as e:
        # В случае ошибки отправляем команду завершения
        print(f"e -1")
        sys.stdout.flush()
        
if __name__ == "__main__":
    main()
//
//  exceptions.h
//  exceptions
//
//  Created by 孙瑞琦 on 2020/1/21.
//

#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

class NotImplementedError: public std::exception {
  public:
    const char * what() const noexcept override {
        return "函数没有实现";
    }
};

#endif /* EXCEPTIONS_H */